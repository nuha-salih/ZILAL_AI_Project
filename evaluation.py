"""
evaluation.py — المقاييس، المقارنة العادلة، التوصيات، وتحدي ZILAL الأخضر (اختياري).

تعريفات المقاييس (كلها نتائج محاكاة وليست قياسات):
• energy_kwh: مجموع (Q_cool/COP)·dt/3.6e6 على كل الخطوات.
• comfort_violation_h: ساعات الإشغال التي كانت فيها T_in خارج [الهدف−هامش، الهدف+هامش].
• violation_fraction: تلك الساعات ÷ ساعات الإشغال (0 إن لم يوجد إشغال).
• unoccupied_energy_kwh: الطاقة في الخطوات التي بلا إشغال.
• unnecessary_energy_kwh: الجزء من الطاقة غير المشغولة الذي *ليس* ضمن نافذة تهيئة
  مسبقة (precond_window_h قبل أقرب إشغال لاحق، افتراضياً 4 ساعات).
• saving_pct = (E_base − E_strategy)/E_base × 100 ؛ الأساس = الاستراتيجية A؛ 0 عند أساس صفري.
• comfort_ok: نسبة المخالفة ≤ 5٪ — التوفير بلا راحة مقبولة لا يُعدّ توفيراً صالحاً.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenarios import SCENARIOS
from strategies import STRATEGY_NAMES_AR, StrategyConfig, make_strategies
from thermal_model import (ORIENTATION_AR, ORIENTATIONS, ClassroomParams, SimInputs,
                           build_sim_inputs, hvac_energy_kwh, simulate)
from weather import slice_weather

COMFORT_OK_FRACTION = 0.05
EPS = 1e-9


def saving_percent(e_baseline: float, e_strategy: float) -> float:
    """((E_base − E_strat)/E_base)×100 مع معالجة آمنة للأساس الصفري."""
    if e_baseline is None or not np.isfinite(e_baseline) or e_baseline <= EPS:
        return 0.0
    return float((e_baseline - e_strategy) / e_baseline * 100.0)


def compute_metrics(df: pd.DataFrame, params: ClassroomParams, inp: SimInputs,
                    precond_window_h: float = 4.0) -> dict:
    dt_h = params.dt_s / 3600.0
    occ = (df["occupancy"].to_numpy() > 0)
    t = df["t_in"].to_numpy()
    hot = occ & (t > params.comfort_high_c + EPS)
    cold = occ & (t < params.comfort_low_c - EPS)
    viol = hot | cold
    occ_h = float(occ.sum() * dt_h)
    e = df["e_hvac_kwh"].to_numpy()
    w = int(precond_window_h * 3600 / params.dt_s)
    nxt = inp.next_occ
    near_future = np.array([(nxt[k] != -1 and nxt[k] - k <= w) for k in range(len(df))])
    unocc = ~occ
    q = df["q_cool_w"].to_numpy()
    return {
        "energy_kwh": float(e.sum()),
        "cooling_thermal_kwh": float((q * params.dt_s / 3.6e6).sum()),
        "solar_gain_kwh": float((df["q_solar_w"].to_numpy() * params.dt_s / 3.6e6).sum()),
        "comfort_violation_h": float(viol.sum() * dt_h),
        "hot_violation_h": float(hot.sum() * dt_h),
        "cold_violation_h": float(cold.sum() * dt_h),
        "occupied_h": occ_h,
        "violation_fraction": float(viol.sum() * dt_h / occ_h) if occ_h > 0 else 0.0,
        "comfort_ok": bool((viol.sum() * dt_h / occ_h) <= COMFORT_OK_FRACTION) if occ_h > 0 else True,
        "unoccupied_energy_kwh": float(e[unocc].sum()),
        "unnecessary_energy_kwh": float(e[unocc & ~near_future].sum()),
        "max_t_occupied_c": float(t[occ].max()) if occ.any() else float("nan"),
        "max_t_c": float(t.max()),
        "peak_cooling_kw": float(q.max() / 1000.0),
    }


def compare_strategies(inp: SimInputs, cfg: StrategyConfig | None = None,
                       precond_window_h: float = 4.0):
    """يشغّل A–D على *نفس* SimInputs (طقس/خصائص/إشغال/مدة/حرارة ابتدائية متطابقة)."""
    strategies = make_strategies(cfg)
    results, rows = {}, {}
    for key, strat in strategies.items():
        df = simulate(inp, strat)
        results[key] = df
        rows[key] = compute_metrics(df, inp.params, inp, precond_window_h)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    base = float(summary.loc["A", "energy_kwh"])
    summary["saving_pct"] = [saving_percent(base, e) for e in summary["energy_kwh"]]
    summary["name_ar"] = [STRATEGY_NAMES_AR[k] for k in summary.index]
    return results, summary


def orientation_shading_sweep(params, weather_slice, occ_blocks, tt_blocks, cfg=None,
                              shadings=(0.0, 0.5, 0.8), strategy_key="C") -> pd.DataFrame:
    """يقيس أثر الاتجاه × التظليل على الكسب الشمسي وطاقة استراتيجية واحدة (افتراضياً C)."""
    rows = []
    for o in ORIENTATIONS:
        for s in sorted(set(round(float(x), 3) for x in shadings)):
            p = params.with_(orientation=o, shading_fraction=s)
            inp = build_sim_inputs(p, weather_slice, occ_blocks, tt_blocks)
            strat = make_strategies(cfg)[strategy_key]
            df = simulate(inp, strat)
            m = compute_metrics(df, p, inp)
            k = int(np.argmax(inp.q_solar))
            rows.append({
                "orientation": o, "shading": s,
                "solar_gain_kwh": m["solar_gain_kwh"],
                "peak_solar_w": float(inp.q_solar.max()),
                "peak_hour": float(inp.hours[k]) if inp.q_solar.max() > 0 else float("nan"),
                "t_out_at_peak_c": float(inp.t_out[k]),
                "energy_kwh": m["energy_kwh"],
                "comfort_violation_h": m["comfort_violation_h"],
            })
    return pd.DataFrame(rows)


def cop_sensitivity(df: pd.DataFrame, dt_s: float, cops=(2.5, 3.0, 3.5)) -> pd.DataFrame:
    """نفس الحمل التبريدي بمعاملات أداء مختلفة (تغيّر COP لا يغيّر الحمل الحراري هنا)."""
    q = df["q_cool_w"].to_numpy()
    return pd.DataFrame({"cop": list(cops),
                         "energy_kwh": [float(hvac_energy_kwh(q, c, dt_s).sum()) for c in cops]})


def hhmm(h: float) -> str:
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}"


def generate_recommendations(summary: pd.DataFrame, sweep: pd.DataFrame, params: ClassroomParams) -> list[str]:
    """توصيات تُولَّد فقط من أرقام محسوبة؛ لا جملة بلا سند."""
    recs: list[str] = []
    a = summary.loc["A"]
    # 1) تشغيل في فترات الخلو
    if a["unoccupied_energy_kwh"] > 0.1 and a["energy_kwh"] > EPS:
        share = a["unoccupied_energy_kwh"] / a["energy_kwh"] * 100
        recs.append(
            f"تشغيل التكييف في الفترات غير المشغولة استهلك {a['unoccupied_energy_kwh']:.1f} kWh "
            f"({share:.0f}٪ من طاقة الاستراتيجية A)، منها {a['unnecessary_energy_kwh']:.1f} kWh خارج نافذة التهيئة المسبقة."
        )
    # 2) أفضل استراتيجية مقبولة الراحة
    cand = summary.drop(index="A")
    ok = cand[(cand["comfort_ok"]) & (cand["saving_pct"] > 0.5)]
    if len(ok):
        best = ok["saving_pct"].idxmax()
        r = ok.loc[best]
        recs.append(
            f"أفضل توفير مع راحة مقبولة: {r['name_ar']} بتوفير محاكى {r['saving_pct']:.0f}٪ "
            f"({r['energy_kwh']:.1f} مقابل {a['energy_kwh']:.1f} kWh للأساس A)."
        )
    for key, r in cand.iterrows():
        if r["saving_pct"] > 0.5 and not r["comfort_ok"]:
            recs.append(
                f"{r['name_ar']} خفّضت الطاقة {r['saving_pct']:.0f}٪ لكن الحرارة خرجت عن نطاق الراحة "
                f"{r['comfort_violation_h']:.1f} ساعة من {r['occupied_h']:.1f} ساعة إشغال؛ لذا لا تُعدّ توفيراً مقبولاً."
            )
    # 3) التنبؤي مقابل القواعد
    c, d = summary.loc["C"], summary.loc["D"]
    if c["occupied_h"] > 0:
        if d["comfort_ok"] and not c["comfort_ok"]:
            recs.append(
                f"التشغيل التنبؤي حافظ على الراحة (مخالفة {d['comfort_violation_h']:.1f} ساعة) بينما سجّل التشغيل "
                f"القائم على القواعد {c['comfort_violation_h']:.1f} ساعة خارج النطاق، لأن التبريد المسبق جهّز القاعة قبل الحضور."
            )
        elif d["comfort_ok"] and c["comfort_ok"] and d["energy_kwh"] < c["energy_kwh"] - 0.05:
            recs.append(
                f"التشغيل التنبؤي حافظ على الراحة بطاقة أقل من القائم على القواعد "
                f"({d['energy_kwh']:.1f} مقابل {c['energy_kwh']:.1f} kWh) في هذه الظروف المحاكاة."
            )
        elif d["comfort_ok"] and c["comfort_ok"]:
            recs.append("لم تُظهر هذه الإعدادات أفضلية طاقية واضحة للتشغيل التنبؤي على القائم على القواعد.")
    # 4) التظليل
    if len(sweep):
        o = params.orientation
        sub = sweep[sweep["orientation"] == o].set_index("shading")
        if 0.0 in sub.index and sub.index.max() > 0:
            hi = sub.index.max()
            s0, s1 = sub.loc[0.0], sub.loc[hi]
            if s0["solar_gain_kwh"] > EPS:
                red = (s0["solar_gain_kwh"] - s1["solar_gain_kwh"]) / s0["solar_gain_kwh"] * 100
                ered = saving_percent(s0["energy_kwh"], s1["energy_kwh"])
                recs.append(
                    f"زيادة التظليل من 0 إلى {hi:.1f} للنافذة {ORIENTATION_AR[o]} خفّضت الكسب الشمسي اليومي "
                    f"بنسبة {red:.0f}٪ ({s0['solar_gain_kwh']:.1f} → {s1['solar_gain_kwh']:.1f} kWh) "
                    f"وطاقة التكييف (الاستراتيجية C) بنسبة {ered:.0f}٪ ({s0['energy_kwh']:.1f} → {s1['energy_kwh']:.1f} kWh)."
                )
        # 5) الاتجاه
        cur = sweep[np.isclose(sweep["shading"], params.shading_fraction)]
        if len(cur) == len(ORIENTATIONS):
            ordered = cur.sort_values("solar_gain_kwh", ascending=False).reset_index(drop=True)
            top, second, low = ordered.iloc[0], ordered.iloc[1], ordered.iloc[-1]
            peaks = "، ".join(f"{ORIENTATION_AR[r.orientation]} عند {hhmm(r.peak_hour)} (حرارة خارجية {r.t_out_at_peak_c:.1f}°م)"
                              for r in cur.itertuples() if np.isfinite(r.peak_hour))
            if top["solar_gain_kwh"] > EPS:
                gap = (top["solar_gain_kwh"] - second["solar_gain_kwh"]) / top["solar_gain_kwh"] * 100
                if gap < 3:
                    lead = (f"النافذتان {ORIENTATION_AR[top['orientation']]} و{ORIENTATION_AR[second['orientation']]} "
                            f"متقاربتان في الكسب الشمسي اليومي ({top['solar_gain_kwh']:.1f} و{second['solar_gain_kwh']:.1f} kWh)")
                else:
                    lead = (f"سجّلت النافذة {ORIENTATION_AR[top['orientation']]} أعلى كسب شمسي ({top['solar_gain_kwh']:.1f} kWh)")
                recs.append(
                    f"بحسب نموذج تعرّض الواجهة المبسّط وبيانات الطقس المختارة: {lead}، وسجّلت {ORIENTATION_AR[low['orientation']]} "
                    f"أدنى كسب ({low['solar_gain_kwh']:.1f} kWh). أوقات ذروة الكسب: {peaks}."
                )
    if not recs:
        recs.append("لا توجد فروق كافية في هذه الإعدادات لتوليد توصيات؛ جرّب سيناريو آخر أو غيّر المدخلات.")
    return recs


def run_analysis(params: ClassroomParams, scenario_key: str, weather_df: pd.DataFrame,
                 start_date, duration_h: int, cfg: StrategyConfig | None = None) -> dict:
    """خط معالجة كامل (بلا واجهة): تحقق → مقارنة عادلة → حساسية → توصيات."""
    params.validate()
    cfg = (cfg or StrategyConfig()).validate()
    if scenario_key not in SCENARIOS:
        raise ValueError("سيناريو غير معروف.")
    sc = SCENARIOS[scenario_key]
    wslice = slice_weather(weather_df, start_date, duration_h)
    inp = build_sim_inputs(params, wslice, sc.occupancy_blocks, sc.timetable_blocks)
    results, summary = compare_strategies(inp, cfg)
    sweep = orientation_shading_sweep(params, wslice, sc.occupancy_blocks, sc.timetable_blocks, cfg,
                                      shadings=(0.0, 0.5, 0.8, params.shading_fraction))
    cop_tbl = {k: cop_sensitivity(df, params.dt_s) for k, df in results.items()}
    # حالة بلا تظليل للمقارنة في الرسم
    inp0 = build_sim_inputs(params.with_(shading_fraction=0.0), wslice, sc.occupancy_blocks, sc.timetable_blocks)
    return {
        "inputs": inp, "results": results, "summary": summary, "sweep": sweep, "cop": cop_tbl,
        "recommendations": generate_recommendations(summary, sweep, params),
        "q_solar_unshaded": inp0.q_solar, "scenario": sc, "weather_slice": wslice,
    }


# ---------------------------------------------------------------------------
# ZILAL Green Challenge (optional, illustrative sample data only)
# ---------------------------------------------------------------------------
GREEN_CHALLENGE_SAMPLE = pd.DataFrame({
    "college": ["كلية الهندسة", "كلية الحاسب", "كلية العلوم"],
    "baseline_kwh": [420.0, 310.0, 365.0],
    "strategy_kwh": [310.0, 262.0, 241.0],
    "comfort_compliance": [0.97, 0.99, 0.78],
})


def green_challenge_ranking(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    نقاط = تحسّن مُطبَّع مقابل خط أساس الكلية نفسها × معامل راحة.
    معامل الراحة: 1 إن كان الالتزام ≥ 0.95، وإلا الالتزام/0.95؛ وصفر إن كان < 0.80.
    ليست الطاقة الخام معياراً للترتيب. بيانات توضيحية فقط.
    """
    d = (GREEN_CHALLENGE_SAMPLE if df is None else df).copy()
    d["improvement_pct"] = [saving_percent(b, s) for b, s in zip(d["baseline_kwh"], d["strategy_kwh"])]
    comp = d["comfort_compliance"].clip(0, 1)
    d["comfort_factor"] = np.where(comp < 0.80, 0.0, np.minimum(1.0, comp / 0.95))
    d["score"] = d["improvement_pct"] * d["comfort_factor"]
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d
