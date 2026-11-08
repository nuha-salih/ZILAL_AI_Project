"""
thermal_model.py — محرك المحاكاة الحرارية (نموذج RC أحادي المنطقة)
===================================================================

المعادلة الحاكمة (توازن طاقة للقاعة كمنطقة واحدة):

    C · dT_in/dt = UA·(T_out − T_in) + Q_solar + Q_internal − Q_cool

التكامل العددي: طريقة أويلر الصريحة بخطوة dt = 900 ثانية (15 دقيقة):

    T_next = T_in + (dt / C) · [UA·(T_out − T_in) + Q_solar + Q_internal − Q_cool]

الاستقرار: الطريقة الصريحة مستقرة عندما dt·UA/C < 2، ويُفرض هنا شرط أشد
(dt·UA/C < 0.5) للحصول على سلوك رتيب بلا تذبذب عددي.

ملاحظة صدق علمي: جميع معاملات القاعة الافتراضية هنا *افتراضات هندسية*
وليست قياسات لقاعة حقيقية في جامعة نجران.

نموذج الواجهة (Orientation) — تقريب مبسط وليس هندسة شمسية تفصيلية:
    G_facade(t) = GHI(t) × f_orientation(hour)
    f(h) = d + (peak − d) · exp(−½((h − h0)/σ)²)
حيث d نسبة الإشعاع المنتشر الذي يراه سطح رأسي، و (peak, h0, σ) تصف ذروة
تعرّض الواجهة (الشرق صباحاً، الغرب عصراً، الجنوب منخفض الذروة صيفاً (0.30) عند
خط عرض ~17.5° شمالاً لأن الشمس شبه عمودية ظهراً). يمكن تجاوز هذا التقريب
بتزويد أعمدة إشعاع مخصصة لكل واجهة في ملف الطقس (g_east_wm2 ...).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

DT_S = 900.0
ORIENTATIONS = ("East", "West", "South")
ORIENTATION_AR = {"East": "شرق", "West": "غرب", "South": "جنوب"}

# (peak, h0, sigma) — افتراضات موثّقة لتقريب تعرّض الواجهة الرأسية
ORIENTATION_PROFILES = {
    "East": {"peak": 1.00, "h0": 8.0, "sigma": 2.0},
    "West": {"peak": 1.00, "h0": 16.0, "sigma": 2.0},
    "South": {"peak": 0.30, "h0": 12.0, "sigma": 3.0},
}
DIFFUSE_FACTOR = 0.15


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassroomParams:
    """معاملات القاعة. كل القيم افتراضية (غير مقاسة) ما لم يُذكر خلاف ذلك."""

    area_m2: float = 60.0
    max_occupancy: int = 30
    occupancy: int = 30  # عدد الطلاب أثناء فترات الإشغال
    window_area_m2: float = 12.0
    orientation: str = "West"
    shading_fraction: float = 0.0
    shgc: float = 0.60
    ua_w_per_k_m2: float = 3.4  # UA لكل م² من مساحة الأرضية → ~204 W/K
    capacitance_j_per_k_m2: float = 150_000.0  # السعة الحرارية الفعالة لكل م²
    q_person_w: float = 75.0
    lighting_w_per_m2: float = 12.0
    equipment_w: float = 300.0
    hvac_capacity_w: float = 15_000.0  # قدرة تبريد حرارية قصوى
    cop: float = 3.0
    comfort_setpoint_c: float = 24.0
    comfort_tolerance_c: float = 1.5
    initial_temp_c: float = 24.0
    dt_s: float = DT_S

    # --- derived -----------------------------------------------------------
    @property
    def ua_w_per_k(self) -> float:
        return self.ua_w_per_k_m2 * self.area_m2

    @property
    def capacitance_j_per_k(self) -> float:
        return self.capacitance_j_per_k_m2 * self.area_m2

    @property
    def effective_occupancy(self) -> int:
        return int(min(self.occupancy, self.max_occupancy))

    @property
    def comfort_low_c(self) -> float:
        return self.comfort_setpoint_c - self.comfort_tolerance_c

    @property
    def comfort_high_c(self) -> float:
        return self.comfort_setpoint_c + self.comfort_tolerance_c

    def with_(self, **kw) -> "ClassroomParams":
        return replace(self, **kw)

    def validate(self) -> "ClassroomParams":
        """يرفع ValueError برسالة واضحة عند أي مدخل غير صالح."""
        def num(name, v, lo, hi):
            if not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
                raise ValueError(f"القيمة '{name}' يجب أن تكون رقماً صالحاً.")
            if not (lo <= v <= hi):
                raise ValueError(f"القيمة '{name}' = {v} خارج المدى المسموح [{lo}, {hi}].")

        num("area_m2", self.area_m2, 10, 500)
        num("max_occupancy", self.max_occupancy, 1, 200)
        num("occupancy", self.occupancy, 0, 200)
        if self.occupancy > self.max_occupancy:
            raise ValueError("عدد الطلاب لا يمكن أن يتجاوز السعة القصوى للقاعة.")
        num("window_area_m2", self.window_area_m2, 0, 100)
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"اتجاه النافذة يجب أن يكون أحد: {ORIENTATIONS}.")
        num("shading_fraction", self.shading_fraction, 0.0, 1.0)
        num("shgc", self.shgc, 0.0, 1.0)
        num("ua_w_per_k_m2", self.ua_w_per_k_m2, 0.1, 30)
        num("capacitance_j_per_k_m2", self.capacitance_j_per_k_m2, 10_000, 2_000_000)
        num("q_person_w", self.q_person_w, 0, 300)
        num("lighting_w_per_m2", self.lighting_w_per_m2, 0, 50)
        num("equipment_w", self.equipment_w, 0, 20_000)
        num("hvac_capacity_w", self.hvac_capacity_w, 100, 200_000)
        num("cop", self.cop, 1.0, 6.0)
        num("comfort_setpoint_c", self.comfort_setpoint_c, 18, 30)
        num("comfort_tolerance_c", self.comfort_tolerance_c, 0.1, 5)
        num("initial_temp_c", self.initial_temp_c, 5, 50)
        num("dt_s", self.dt_s, 60, 3600)
        ratio = self.dt_s * self.ua_w_per_k / self.capacitance_j_per_k
        if ratio >= 0.5:
            raise ValueError(
                f"الإعدادات غير مستقرة عددياً: dt·UA/C = {ratio:.2f} ≥ 0.5. "
                "قلّل UA أو زد السعة الحرارية C."
            )
        return self


# ----------------------------------------------------------------------------
# Physics building blocks
# ----------------------------------------------------------------------------
def facade_irradiance(orientation: str, hours: np.ndarray, ghi: np.ndarray) -> np.ndarray:
    """تقريب إشعاع الواجهة الرأسية G_facade من GHI وملف تعرّض زمني (انظر أعلى الملف)."""
    if orientation not in ORIENTATION_PROFILES:
        raise ValueError(f"اتجاه غير معروف: {orientation}")
    prof = ORIENTATION_PROFILES[orientation]
    d = DIFFUSE_FACTOR
    h = np.asarray(hours, dtype=float)
    f = d + (prof["peak"] - d) * np.exp(-0.5 * ((h - prof["h0"]) / prof["sigma"]) ** 2)
    return np.asarray(ghi, dtype=float) * f


def solar_gain(window_area_m2, g_facade, shgc, shading_fraction):
    """Q_solar = A_window · G_orientation · SHGC · (1 − shading_fraction)  [W]"""
    return window_area_m2 * np.asarray(g_facade, dtype=float) * shgc * (1.0 - shading_fraction)


def internal_gain(params: ClassroomParams, n_students):
    """
    Q_internal = N·q_person + P_lighting + P_equipment  [W]
    الإضاءة والأجهزة تُشغَّل فقط عند وجود إشغال (N > 0) — افتراض موثّق.
    """
    n = np.asarray(n_students, dtype=float)
    on = (n > 0).astype(float)
    return (n * params.q_person_w
            + on * (params.lighting_w_per_m2 * params.area_m2 + params.equipment_w))


def step_temperature(t_in, t_out, q_solar, q_internal, q_cool, c, ua, dt=DT_S):
    """خطوة أويلر الصريحة لمعادلة RC."""
    return t_in + (dt / c) * (ua * (t_out - t_in) + q_solar + q_internal - q_cool)


def cooling_to_target(t_in, target, t_out, q_solar, q_internal, c, ua, dt=DT_S):
    """قدرة التبريد التي تجعل T_next = target بالضبط (قد تكون سالبة = لا حاجة للتبريد)."""
    return ua * (t_out - t_in) + q_solar + q_internal + (c / dt) * (t_in - target)


def clip_cooling(q, capacity_w):
    """يمنع التبريد السالب وتجاوز قدرة المكيّف."""
    return float(min(max(q, 0.0), capacity_w))


def hvac_energy_kwh(q_cool_w, cop, dt=DT_S):
    """E_HVAC [kWh] = (Q_cool / COP) · dt / 3 600 000"""
    return (np.asarray(q_cool_w, dtype=float) / cop) * dt / 3_600_000.0


# ----------------------------------------------------------------------------
# Simulation inputs / loop
# ----------------------------------------------------------------------------
def in_blocks(hours: np.ndarray, blocks: Iterable[Sequence[float]]) -> np.ndarray:
    """قناع منطقي: هل الساعة داخل أي فترة (start ≤ h < end)."""
    h = np.asarray(hours, dtype=float)
    mask = np.zeros(h.shape, dtype=bool)
    for s, e in blocks:
        mask |= (h >= s) & (h < e)
    return mask


@dataclass
class SimInputs:
    """مدخلات موحّدة تُمرَّر كما هي لكل الاستراتيجيات (ضمان عدالة المقارنة)."""
    params: ClassroomParams
    timestamps: np.ndarray
    hours: np.ndarray
    t_out: np.ndarray
    g_facade: np.ndarray
    q_solar: np.ndarray
    q_internal: np.ndarray
    occ: np.ndarray  # عدد الطلاب الفعلي في كل خطوة
    timetable_on: np.ndarray  # الجدول المعلن (قد يختلف عن الإشغال الفعلي)
    next_occ: np.ndarray  # فهرس أول خطوة مشغولة بعد k، أو −1

    @property
    def n(self) -> int:
        return len(self.t_out)


def _next_occupied_index(occ: np.ndarray) -> np.ndarray:
    n = len(occ)
    nxt = np.full(n, -1, dtype=int)
    upcoming = -1
    for k in range(n - 1, -1, -1):
        nxt[k] = upcoming
        if occ[k] > 0:
            upcoming = k
    return nxt


def build_sim_inputs(params: ClassroomParams, weather, occupancy_blocks, timetable_blocks) -> SimInputs:
    """
    weather: DataFrame بأعمدة timestamp, outdoor_temp_c, solar_irradiance_wm2
    (وأعمدة اختيارية g_east_wm2 / g_west_wm2 / g_south_wm2 لتجاوز التقريب).
    """
    params.validate()
    ts = weather["timestamp"].dt
    hours = (ts.hour + ts.minute / 60.0).to_numpy(dtype=float)
    t_out = weather["outdoor_temp_c"].to_numpy(dtype=float)
    ghi = weather["solar_irradiance_wm2"].to_numpy(dtype=float)
    col = f"g_{params.orientation.lower()}_wm2"
    if col in weather.columns:
        g_fac = weather[col].to_numpy(dtype=float)
    else:
        g_fac = facade_irradiance(params.orientation, hours, ghi)
    q_sol = solar_gain(params.window_area_m2, g_fac, params.shgc, params.shading_fraction)
    occ = np.where(in_blocks(hours, occupancy_blocks), params.effective_occupancy, 0)
    q_int = internal_gain(params, occ)
    return SimInputs(
        params=params,
        timestamps=weather["timestamp"].to_numpy(),
        hours=hours, t_out=t_out, g_facade=g_fac, q_solar=q_sol, q_internal=q_int,
        occ=occ, timetable_on=in_blocks(hours, timetable_blocks),
        next_occ=_next_occupied_index(occ),
    )


def simulate(inp: SimInputs, strategy):
    """
    يشغّل المحاكاة. strategy يوفّر reset(inp) و decide(k, T, inp) → طلب تبريد [W]،
    ثم يُقيَّد الطلب بين 0 وقدرة المكيّف.
    يعيد: dict من المصفوفات (t_in هي درجة الحرارة في بداية كل خطوة).
    """
    import pandas as pd

    p = inp.params
    n = inp.n
    c, ua, dt = p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s
    strategy.reset(inp)
    t_in = np.empty(n)
    q_cool = np.empty(n)
    T = float(p.initial_temp_c)
    for k in range(n):
        t_in[k] = T
        req = float(strategy.decide(k, T, inp))
        if not np.isfinite(req):
            req = 0.0
        q = clip_cooling(req, p.hvac_capacity_w)
        q_cool[k] = q
        T = step_temperature(T, inp.t_out[k], inp.q_solar[k], inp.q_internal[k], q, c, ua, dt)
    return pd.DataFrame({
        "timestamp": inp.timestamps,
        "hour": inp.hours,
        "t_out": inp.t_out,
        "g_facade": inp.g_facade,
        "q_solar_w": inp.q_solar,
        "q_internal_w": inp.q_internal,
        "occupancy": inp.occ,
        "t_in": t_in,
        "q_cool_w": q_cool,
        "e_hvac_kwh": hvac_energy_kwh(q_cool, p.cop, dt),
    })
