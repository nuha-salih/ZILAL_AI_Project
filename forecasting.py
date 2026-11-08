"""
forecasting.py — التنبؤ الحراري.

1) تنبؤ فيزيائي (يستخدمه التحكم التنبؤي D): دحرجة نموذج RC للأمام بتوقعات الطقس
   والإشغال. في هذا النموذج الأولي يُفترض أن التوقعات مطابقة للبيانات (توقع مثالي) —
   وهذا قيد موثّق، فالتوقعات الحقيقية فيها أخطاء.
2) نموذج تعلّم آلي اختياري (Random Forest واحد فقط): يتعلم من بيانات *ولّدها المحاكي نفسه*؛
   لذلك دقته تقيس قدرته على تقليد المحاكي فقط، وليست دقة في العالم الحقيقي. لا يُستخدم في
   التحكم، ويُعرض منفصلاً عن مخرجات المحاكاة الفيزيائية.
"""
from __future__ import annotations

import numpy as np

from thermal_model import ClassroomParams, SimInputs, step_temperature


def free_float_forecast(t0: float, k: int, j: int, inp: SimInputs) -> float:
    """درجة الحرارة المتوقعة في بداية الخطوة j لو لم يعمل التبريد من الخطوة k (فيزيائي)."""
    p = inp.params
    c, ua, dt = p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s
    t = t0
    for i in range(k, j):
        t = step_temperature(t, inp.t_out[i], inp.q_solar[i], inp.q_internal[i], 0.0, c, ua, dt)
    return t


def required_cooling_for_arrival(t0: float, k: int, j: int, target: float, inp: SimInputs) -> float:
    """
    قدرة تبريد ثابتة Q (W) من الخطوة k حتى j−1 تجعل T_j = target.
    النموذج خطي، لذا: T_j = T_free_j − (dt/C)·Q·Σ_{m=0}^{n−1}(1−a)^m ، a = dt·UA/C.
    قيمة موجبة = يلزم تبريد؛ سالبة/صفر = القاعة ستبقى ضمن الهدف دون تبريد.
    """
    p = inp.params
    c, ua, dt = p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s
    n = j - k
    if n <= 0:
        return 0.0
    a = dt * ua / c
    s = (1.0 - (1.0 - a) ** n) / a
    t_free = free_float_forecast(t0, k, j, inp)
    return (t_free - target) * c / (dt * s)


# ---------------------------------------------------------------------------
# Optional ML surrogate (Random Forest) — trained on SIMULATOR-generated data
# ---------------------------------------------------------------------------
def train_surrogate(params: ClassroomParams, weather, n_samples: int = 4000, seed: int = 0) -> dict:
    """
    يدرّب Random Forest واحداً للتنبؤ بتغيّر الحرارة خلال خطوة واحدة (T_next − T_in) من
    [T_in, T_out, Q_solar, Q_internal, Q_cool, hour_sin, hour_cos].
    الأمثلة مولَّدة من محاكي RC نفسه، فالنتيجة تقيس «تقليد المحاكي» فقط.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    from thermal_model import facade_irradiance, internal_gain, solar_gain

    params.validate()
    rng = np.random.default_rng(seed)
    ts = weather["timestamp"].dt
    hours = (ts.hour + ts.minute / 60.0).to_numpy(dtype=float)
    t_out_all = weather["outdoor_temp_c"].to_numpy(dtype=float)
    ghi = weather["solar_irradiance_wm2"].to_numpy(dtype=float)
    g_fac = facade_irradiance(params.orientation, hours, ghi)
    q_sol_all = solar_gain(params.window_area_m2, g_fac, params.shgc, params.shading_fraction)

    idx = rng.integers(0, len(hours), n_samples)
    t_in = rng.uniform(20.0, 36.0, n_samples)
    n_stu = rng.integers(0, params.max_occupancy + 1, n_samples)
    q_int = internal_gain(params, n_stu)
    q_cool = rng.uniform(0.0, params.hvac_capacity_w, n_samples)
    t_out, q_sol, h = t_out_all[idx], q_sol_all[idx], hours[idx]
    t_next = step_temperature(t_in, t_out, q_sol, q_int, q_cool,
                              params.capacitance_j_per_k, params.ua_w_per_k, params.dt_s)
    X = np.column_stack([t_in, t_out, q_sol, q_int, q_cool,
                         np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])
    y = t_next - t_in
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed, n_jobs=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "model": model,
        "mae_k_per_step": float(np.mean(np.abs(pred - yte))),
        "target_std_k_per_step": float(np.std(yte)),
        "n_train": len(ytr), "n_test": len(yte),
        "note": "بيانات التدريب والاختبار مولَّدة من المحاكي نفسه؛ التحقق في العالم الحقيقي عمل مستقبلي.",
    }
