"""
ZILAL AI | ظِلال — واجهة Streamlit التفاعلية.
شغّل:  streamlit run app.py

تنبيه: كل الأرقام نتائج محاكاة بمعاملات افتراضية وطقس اصطناعي (ما لم تُرفع بيانات خاصة بك)،
وليست قياسات أو وفورات فعلية لجامعة نجران، والنظام لا يتحكم في أي مكيّف حقيقي.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import evaluation as ev
from forecasting import train_surrogate
from scenarios import SCENARIOS
from strategies import STRATEGY_NAMES_AR, StrategyConfig
from thermal_model import ORIENTATION_AR, ORIENTATIONS, ClassroomParams
from weather import (WeatherDataError, generate_synthetic_weather, load_weather_csv,
                     slice_weather, start_options)

# مفتاح تعطيل القسم الاختياري: غيّر إلى False (أو متغير بيئة ZILAL_GREEN=0) دون التأثير على المحاكاة.
ENABLE_GREEN_CHALLENGE = os.environ.get("ZILAL_GREEN", "1") != "0"

GREEN, SAND, SLATE, INK = "#1F5C45", "#B8873B", "#2F6F8F", "#23302A"
STRAT_COLORS = {"A": "#7A7468", "B": SAND, "C": SLATE, "D": GREEN}

def _stretch_kwargs() -> dict:
    """use_container_width تم إهماله في Streamlit الحديث لصالح width="stretch"؛ نختار حسب الإصدار."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()

st.set_page_config(page_title="ظِلال | ZILAL AI", page_icon="🌿", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=Amiri:wght@700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
}}
.stApp {{ background: #EFEBE2; color: {INK}; }}
.block-container {{ direction: rtl; text-align: right; max-width: 1200px; padding-top: 1.2rem; }}
div[data-baseweb="slider"], .js-plotly-plot, .stDataFrame {{ direction: ltr; }}
.zl-head {{
  background: {GREEN}; color: #F6F2E8; border-radius: 6px; padding: 1.6rem 2rem; margin-bottom: 1rem;
  background-image: linear-gradient(115deg, rgba(0,0,0,0) 62%, rgba(0,0,0,.22) 62%);
}}
.zl-head .mark {{ font-family: 'Amiri', serif; font-size: 3rem; line-height: 1.2; }}
.zl-head .mark span {{ font-family: 'IBM Plex Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500;
  letter-spacing: .08em; margin-inline-start: .8rem; opacity: .85; }}
.zl-head .tag {{ font-size: 1.05rem; opacity: .92; margin-top: .2rem; }}
.zl-note {{ border-inline-start: 4px solid {SAND}; background: #E5DFD0; padding: .7rem 1rem; border-radius: 4px;
  font-size: .93rem; margin: .5rem 0 1rem; }}
.zl-tag {{ display:inline-block; padding: .05rem .5rem; border-radius: 3px; font-size: .78rem; margin-inline-end:.3rem;
  background:#D9D2BF; }}
div[data-testid="stMetric"] {{ background: #F7F4EC; border: 1px solid #D9D2BF; padding: .7rem .9rem; border-radius: 6px; }}
button[role="tab"] {{ font-weight: 500; }}
@media (max-width: 700px) {{ .zl-head .mark {{ font-size: 2.2rem; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="zl-head">
  <div class="mark">ظِلال | <span>ZILAL AI</span></div>
  <div class="tag">نتنبأ بالاحتياج، نقلل الهدر، ونحافظ على الراحة.</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def synthetic_weather() -> pd.DataFrame:
    return generate_synthetic_weather()


tab_names = ["لوحة المعلومات", "إعداد القاعة", "الطقس والسيناريو", "نتائج المحاكاة", "التوصيات"]
if ENABLE_GREEN_CHALLENGE:
    tab_names.append("تحدي ZILAL الأخضر")
tab_names.append("الحدود والافتراضات")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ---------------------------------------------------------------------------
# إعداد القاعة
# ---------------------------------------------------------------------------
with tabs["إعداد القاعة"]:
    st.markdown('<div class="zl-note">كل معاملات القاعة هنا <b>افتراضات هندسية غير مقاسة</b> وليست خصائص '
                'قاعة فعلية في جامعة نجران. عدّلها لاستكشاف الأثر.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.number_input("مساحة القاعة (م²)", 20.0, 150.0, 60.0, 5.0)
        max_occ = st.number_input("السعة القصوى (طالب)", 10, 60, 30, 1)
        occupancy = st.slider("عدد الطلاب أثناء المحاضرات", 0, int(max_occ), min(30, int(max_occ)))
    with c2:
        orientation = st.selectbox("اتجاه النافذة", ORIENTATIONS, index=1, format_func=lambda o: f"{ORIENTATION_AR[o]} ({o})")
        win_area = st.slider("مساحة النوافذ (م²)", 0.0, 30.0, 12.0, 0.5)
        shading = st.slider("نسبة التظليل (0 = بلا تظليل، 1 = كامل)", 0.0, 1.0, 0.0, 0.05)
    with c3:
        setpoint = st.slider("درجة الراحة المستهدفة (°م)", 20.0, 28.0, 24.0, 0.5)
        tol = st.slider("هامش الراحة ± (°م)", 0.5, 3.0, 1.5, 0.25)
        cop = st.slider("معامل الأداء COP", 2.5, 3.5, 3.0, 0.1)
        cap_kw = st.slider("قدرة التبريد القصوى (kW حراري)", 5.0, 40.0, 15.0, 0.5)
    with st.expander("معاملات متقدمة (افتراضية — غير مقاسة)"):
        a1, a2, a3 = st.columns(3)
        shgc = a1.slider("SHGC", 0.1, 0.9, 0.60, 0.05)
        ua_m2 = a1.slider("UA لكل م² أرضية (W/K·م²)", 1.0, 8.0, 3.4, 0.1)
        cap_m2 = a2.slider("السعة الحرارية الفعالة (kJ/K·م²)", 50, 400, 150, 10)
        q_person = a2.slider("حرارة الشخص (W)", 60, 120, 75, 5)
        light = a3.slider("الإضاءة (W/م²)", 4.0, 25.0, 12.0, 1.0)
        equip = a3.slider("الأجهزة (W)", 0, 1500, 300, 50)
        t0 = a3.slider("درجة الحرارة الابتدائية (°م)", 18.0, 32.0, 24.0, 0.5)
    params = ClassroomParams(
        area_m2=area, max_occupancy=int(max_occ), occupancy=int(occupancy), window_area_m2=win_area,
        orientation=orientation, shading_fraction=shading, shgc=shgc, ua_w_per_k_m2=ua_m2,
        capacitance_j_per_k_m2=cap_m2 * 1000.0, q_person_w=float(q_person), lighting_w_per_m2=light,
        equipment_w=float(equip), hvac_capacity_w=cap_kw * 1000.0, cop=cop, comfort_setpoint_c=setpoint,
        comfort_tolerance_c=tol, initial_temp_c=t0,
    )

# ---------------------------------------------------------------------------
# الطقس والسيناريو
# ---------------------------------------------------------------------------
with tabs["الطقس والسيناريو"]:
    s1, s2 = st.columns([1, 1])
    with s1:
        scenario_key = st.radio("السيناريو", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].name_ar)
        st.caption(SCENARIOS[scenario_key].description_ar)
        duration_h = st.select_slider("مدة المحاكاة (ساعة)", options=[24, 48, 72], value=24)
    with s2:
        source = st.radio("مصدر الطقس", ["بيانات اصطناعية مُحاكاة (مرفقة)", "رفع ملف CSV"])
        uploaded = None
        if source == "رفع ملف CSV":
            uploaded = st.file_uploader("ملف CSV بالأعمدة: timestamp, outdoor_temp_c, solar_irradiance_wm2", type=["csv"])
    weather_df, weather_label = synthetic_weather(), "اصطناعي مُحاكى"
    if uploaded is not None:
        try:
            weather_df, wwarn = load_weather_csv(uploaded)
            weather_label = "مرفوع من المستخدم"
            for m in wwarn:
                st.warning(m)
        except WeatherDataError as exc:
            st.error(f"تعذّر استخدام الملف: {exc} — سيُستخدم الطقس الاصطناعي بدلاً منه.")
    elif source == "رفع ملف CSV":
        st.info("لم يُرفع ملف بعد؛ يُستخدم الطقس الاصطناعي مؤقتاً.")
    options = start_options(weather_df, duration_h)
    start_date = None
    if not options:
        st.error("لا توجد أيام تغطي مدة المحاكاة المطلوبة في بيانات الطقس. قلّل المدة أو ارفع بيانات أطول.")
    else:
        start_date = st.selectbox("يوم البداية (00:00)", options)
    if source.startswith("بيانات"):
        st.markdown('<div class="zl-note">⚠️ بيانات الطقس هذه <b>اصطناعية (مُحاكاة)</b> لطقس صيفي حار، '
                    'وليست قياسات فعلية لنجران.</div>', unsafe_allow_html=True)
    with st.expander("إعدادات الاستراتيجيات"):
        g1, g2, g3 = st.columns(3)
        fixed_on = g1.slider("A: بداية التشغيل الثابت (ساعة)", 0.0, 12.0, 6.0, 0.5)
        fixed_off = g1.slider("A: نهاية التشغيل الثابت (ساعة)", 12.0, 24.0, 18.0, 0.5)
        deadband = g2.slider("C: هامش بدء التبريد فوق الهدف (°م)", 0.0, 3.0, 0.5, 0.25)
        unocc_lim = g2.slider("C/D: حد وقائي عند خلو القاعة (°م)", 26.0, 38.0, 32.0, 0.5)
        horizon = g3.slider("D: أفق التنبؤ (ساعة)", 1.0, 12.0, 8.0, 0.5)
        offset = g3.slider("D: رفع الهدف أثناء الإشغال (°م)", 0.0, 2.0, 0.5, 0.25)
    cfg = StrategyConfig(fixed_on_h=fixed_on, fixed_off_h=fixed_off, deadband_c=deadband,
                         unocc_limit_c=unocc_lim, horizon_h=horizon, predictive_offset_c=offset)
    if start_date is not None:
        try:
            preview = slice_weather(weather_df, start_date, duration_h)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=preview["timestamp"], y=preview["outdoor_temp_c"], name="درجة الحرارة الخارجية (°م)",
                                     line=dict(color=SAND)), secondary_y=False)
            fig.add_trace(go.Scatter(x=preview["timestamp"], y=preview["solar_irradiance_wm2"], name="الإشعاع الأفقي (W/m²)",
                                     line=dict(color=SLATE, dash="dot")), secondary_y=True)
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title=f"مدخلات الطقس ({weather_label})",
                              legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, **STRETCH)
            with st.expander("عرض جدول الطقس"):
                st.dataframe(preview, **STRETCH, height=250)
        except WeatherDataError as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# تشغيل المحاكاة (تفاعلي: يُعاد الحساب عند أي تغيير في المدخلات)
# ---------------------------------------------------------------------------
analysis = None
error_msg = None
if start_date is None:
    error_msg = "اختر بيانات طقس ويوم بداية صالحين في تبويب «الطقس والسيناريو»."
else:
    try:
        analysis = ev.run_analysis(params, scenario_key, weather_df, start_date, duration_h, cfg)
    except (ValueError, WeatherDataError) as exc:
        error_msg = f"مدخلات غير صالحة: {exc}"


def need_results(tab):
    with tab:
        st.error(error_msg or "لا توجد نتائج.")


def occupied_spans(inp):
    spans, start = [], None
    for k, o in enumerate(inp.occ):
        if o > 0 and start is None:
            start = k
        if (o == 0 or k == len(inp.occ) - 1) and start is not None:
            end = k if o == 0 else k
            spans.append((inp.timestamps[start], inp.timestamps[end]))
            start = None
    return spans


def base_layout(fig, title, height=380, ytitle=None):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans Arabic, Segoe UI, Tahoma"))
    if ytitle:
        fig.update_yaxes(title=ytitle)
    return fig


def shade(fig, inp, row=None):
    for s, e in occupied_spans(inp):
        kw = dict(row=row, col=1) if row else {}
        fig.add_vrect(x0=s, x1=e, fillcolor=GREEN, opacity=0.08, line_width=0, **kw)


def comfort_band(fig, p):
    fig.add_hrect(y0=p.comfort_low_c, y1=p.comfort_high_c, fillcolor=GREEN, opacity=0.10, line_width=0)


# ---------------------------------------------------------------------------
# لوحة المعلومات
# ---------------------------------------------------------------------------
with tabs["لوحة المعلومات"]:
    st.markdown("**ظِلال** منصة محاكاة تتنبأ بالحرارة داخل القاعات الدراسية، وتقيّم الكسب الشمسي وأثر اتجاه النوافذ والتظليل، "
                "وتقارن أربع استراتيجيات لتشغيل التكييف من حيث الطاقة والراحة الحرارية.")
    st.markdown('<div class="zl-note"><b>حالة المحاكاة:</b> نتائج <b>محاكاة</b> بنموذج RC مبسّط، '
                f'الطقس: <b>{weather_label}</b>، ومعاملات القاعة <b>افتراضية غير مقاسة</b>. '
                'لا يتحكم النظام حالياً بأي مكيّف فعلي، والتوفير المعروض تقدير محاكى وليس وفراً فعلياً.</div>',
                unsafe_allow_html=True)
    if analysis is None:
        st.error(error_msg)
    else:
        sm = analysis["summary"]
        ok = sm.drop(index="A")
        ok = ok[(ok["comfort_ok"]) & (ok["saving_pct"] > 0)]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("طاقة الأساس A (kWh)", f"{sm.loc['A', 'energy_kwh']:.1f}")
        if len(ok):
            b = ok["saving_pct"].idxmax()
            k2.metric("أفضل توفير بشرط الراحة (محاكى)", f"{ok.loc[b, 'saving_pct']:.0f}٪", help=STRATEGY_NAMES_AR[b])
            k3.metric("الاستراتيجية", STRATEGY_NAMES_AR[b].split(" — ")[1])
        else:
            k2.metric("أفضل توفير بشرط الراحة (محاكى)", "—", help="لا توجد استراتيجية توفّر طاقة مع راحة مقبولة.")
            k3.metric("الاستراتيجية", "—")
        k4.metric("تشغيل A في فترات الخلو (kWh)", f"{sm.loc['A', 'unoccupied_energy_kwh']:.1f}")
        st.caption(f"السيناريو: {analysis['scenario'].name_ar} — {duration_h} ساعة — اتجاه النافذة: "
                   f"{ORIENTATION_AR[params.orientation]} — التظليل: {params.shading_fraction:.2f}")

# ---------------------------------------------------------------------------
# النتائج
# ---------------------------------------------------------------------------
if analysis is None:
    need_results(tabs["نتائج المحاكاة"])
    need_results(tabs["التوصيات"])
else:
    inp, res, sm = analysis["inputs"], analysis["results"], analysis["summary"]
    with tabs["نتائج المحاكاة"]:
        st.markdown('<span class="zl-tag">مخرجات فيزيائية محاكاة</span>', unsafe_allow_html=True)
        pick = st.selectbox("الاستراتيجية في الرسم الأول", list(res), index=3, format_func=lambda k: STRATEGY_NAMES_AR[k])
        # 1) داخلي مقابل خارجي
        d = res[pick]
        fig = go.Figure()
        comfort_band(fig, params); shade(fig, inp)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["t_in"], name="داخلية", line=dict(color=GREEN, width=3)))
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["t_out"], name="خارجية", line=dict(color=SAND, dash="dot")))
        st.plotly_chart(base_layout(fig, "١) درجة الحرارة الداخلية مقابل الخارجية (المنطقة الخضراء = نطاق الراحة، التظليل = إشغال)",
                                    ytitle="°م"), **STRETCH)
        # 2) عبر الاستراتيجيات
        fig = go.Figure(); comfort_band(fig, params); shade(fig, inp)
        for k, df in res.items():
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["t_in"], name=STRATEGY_NAMES_AR[k], line=dict(color=STRAT_COLORS[k])))
        st.plotly_chart(base_layout(fig, "٢) الحرارة الداخلية عبر الاستراتيجيات", ytitle="°م"), **STRETCH)
        # 3) الطاقة
        fig = go.Figure(go.Bar(x=[STRATEGY_NAMES_AR[k] for k in sm.index], y=sm["energy_kwh"],
                               marker_color=[STRAT_COLORS[k] for k in sm.index],
                               text=[f"{v:.1f}" for v in sm["energy_kwh"]], textposition="outside"))
        st.plotly_chart(base_layout(fig, "٣) استهلاك طاقة التكييف الكهربائية (kWh)", 340, "kWh"), **STRETCH)
        # 4) الكسب الشمسي
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["q_solar_w"] / 1000, name=f"الكسب الشمسي (تظليل {params.shading_fraction:.2f})",
                                 line=dict(color=SAND, width=3), fill="tozeroy"))
        if params.shading_fraction > 0:
            fig.add_trace(go.Scatter(x=d["timestamp"], y=analysis["q_solar_unshaded"] / 1000, name="بلا تظليل",
                                     line=dict(color="#7A7468", dash="dash")))
        st.plotly_chart(base_layout(fig, f"٤) الكسب الشمسي عبر الزجاج — واجهة {ORIENTATION_AR[params.orientation]}", 340, "kW"),
                        **STRETCH)
        st.caption("إشعاع الواجهة تقريب مبسّط: GHI × ملف تعرّض زمني للاتجاه (وليس محرك هندسة شمسية تفصيلياً).")
        # 5) تشغيل التكييف
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=[STRATEGY_NAMES_AR[k] for k in res])
        for i, (k, df) in enumerate(res.items(), start=1):
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["q_cool_w"] / 1000, line=dict(color=STRAT_COLORS[k]),
                                     fill="tozeroy", showlegend=False), row=i, col=1)
            shade(fig, inp, row=i)
        fig.update_yaxes(title_text="kW")
        st.plotly_chart(base_layout(fig, "٥) قدرة التبريد (kW حراري) عبر الزمن", 640), **STRETCH)

        st.subheader("جدول المقاييس")
        show = sm[["name_ar", "energy_kwh", "saving_pct", "comfort_violation_h", "violation_fraction", "comfort_ok",
                   "unoccupied_energy_kwh", "unnecessary_energy_kwh", "max_t_c", "peak_cooling_kw"]].copy()
        show["violation_fraction"] = (show["violation_fraction"] * 100).round(1)
        show.columns = ["الاستراتيجية", "الطاقة (kWh)", "التوفير مقابل A (٪)", "ساعات خارج الراحة (أثناء الإشغال)",
                        "نسبة المخالفة (٪)", "الراحة مقبولة (≤5٪)", "طاقة وقت الخلو (kWh)", "طاقة غير ضرورية (kWh)",
                        "أعلى حرارة (°م)", "ذروة التبريد (kW)"]
        st.dataframe(show.round(2), **STRETCH, hide_index=True)
        with st.expander("كيف تُحسب المقاييس؟"):
            st.markdown("""
- **الطاقة** = Σ (Q_cool ÷ COP) × dt ÷ 3 600 000، مع قيد 0 ≤ Q_cool ≤ قدرة المكيّف.
- **ساعات خارج الراحة**: خطوات الإشغال التي كانت فيها الحرارة خارج [الهدف − الهامش، الهدف + الهامش] (حرارة أو برودة زائدة). لا تُحتسب أثناء خلو القاعة.
- **التوفير** = (E_A − E_الاستراتيجية) ÷ E_A × 100 مع الأساس = A؛ يُعامَل الأساس الصفري بأمان. **توفير بلا راحة مقبولة (> 5٪ مخالفة) لا يُعدّ توفيراً صالحاً.**
- **طاقة وقت الخلو**: الطاقة في خطوات بلا إشغال. **غير الضرورية**: الجزء الواقع خارج نافذة تهيئة مسبقة مدتها 4 ساعات قبل أقرب إشغال لاحق.
- كل الاستراتيجيات تُشغَّل على نفس الطقس والقاعة والحرارة الابتدائية والإشغال والمدة.
- **أعلى حرارة** تشمل أوقات الخلو (تُظهر تراكم الحرارة عند إيقاف التبريد).
""")
        st.subheader("حساسية الاتجاه × التظليل (الاستراتيجية C)")
        sw = analysis["sweep"].copy()
        sw["orientation"] = sw["orientation"].map(ORIENTATION_AR)
        sw["peak_hour"] = sw["peak_hour"].map(lambda h: ev.hhmm(h) if np.isfinite(h) else "—")
        sw.columns = ["الاتجاه", "التظليل", "الكسب الشمسي (kWh)", "ذروة الكسب (W)", "وقت الذروة", "حرارة خارجية عند الذروة (°م)",
                      "طاقة التكييف (kWh)", "ساعات خارج الراحة"]
        st.dataframe(sw.round(2), **STRETCH, hide_index=True)
        st.subheader("حساسية COP (نفس الحمل التبريدي)")
        cop_choice = st.selectbox("الاستراتيجية", list(res), index=3, format_func=lambda k: STRATEGY_NAMES_AR[k], key="copsel")
        ct = analysis["cop"][cop_choice].copy()
        ct.columns = ["COP", "الطاقة (kWh)"]
        st.dataframe(ct.round(2), **STRETCH, hide_index=True)

        with st.expander("نموذج تعلّم آلي تجريبي (Random Forest) — اختياري ولا يتحكم بأي شيء"):
            st.markdown('<span class="zl-tag">تنبؤ تعلّم آلي</span> يُدرَّب على بيانات <b>ولّدها المحاكي نفسه</b>، '
                        'لذا يقيس قدرته على تقليد المحاكي فقط وليس دقة في العالم الحقيقي. التحقق الفعلي عمل مستقبلي. '
                        'التحكم التنبؤي D يعتمد على التنبؤ الفيزيائي بنموذج RC لا على هذا النموذج.', unsafe_allow_html=True)
            if st.button("درّب النموذج التجريبي"):
                try:
                    with st.spinner("جارٍ التدريب..."):
                        st.session_state["rf"] = train_surrogate(params, analysis["weather_slice"], n_samples=4000)
                except ImportError as exc:
                    st.error(f"scikit-learn غير متاح في هذه البيئة ({exc}). هذه الميزة اختيارية؛ بقية التطبيق لا يتأثر.")
            rf = st.session_state.get("rf")
            if rf:
                m1, m2 = st.columns(2)
                m1.metric("متوسط الخطأ المطلق (°م لكل خطوة 15 د)", f"{rf['mae_k_per_step']:.4f}")
                m2.metric("الانحراف المعياري للهدف (°م/خطوة)", f"{rf['target_std_k_per_step']:.3f}")
                st.caption(f"{rf['n_train']} عينة تدريب / {rf['n_test']} اختبار — {rf['note']}")

    with tabs["التوصيات"]:
        st.caption("كل توصية مولَّدة من أرقام هذه المحاكاة تحديداً؛ تتغير عند تغيير المدخلات، وهي ليست وفورات فعلية.")
        for r in analysis["recommendations"]:
            st.markdown(f"- {r}")

# ---------------------------------------------------------------------------
# تحدي ZILAL الأخضر (اختياري، مستقل تماماً عن المحاكاة)
# ---------------------------------------------------------------------------
if ENABLE_GREEN_CHALLENGE:
    with tabs["تحدي ZILAL الأخضر"]:
        st.markdown('<div class="zl-note"><b>بيانات توضيحية فقط</b> — ليست قياسات فعلية لأي كلية. '
                    'الترتيب = التحسّن المُطبَّع مقابل خط أساس كل كلية × معامل راحة '
                    '(1 إن كان الالتزام ≥ 95٪، يتناقص بعدها، ويُصفَّر دون 80٪). الطاقة الخام ليست معيار الترتيب.</div>',
                    unsafe_allow_html=True)
        lb = ev.green_challenge_ranking()
        view = lb[["rank", "college", "baseline_kwh", "strategy_kwh", "improvement_pct", "comfort_compliance", "score"]].copy()
        view["comfort_compliance"] = (view["comfort_compliance"] * 100).round(0)
        view.columns = ["الترتيب", "الكلية", "الأساس (kWh)", "بعد الاستراتيجية (kWh)", "التحسّن (٪)", "الالتزام بالراحة (٪)", "النقاط"]
        st.dataframe(view.round(1), **STRETCH, hide_index=True)

# ---------------------------------------------------------------------------
# الحدود
# ---------------------------------------------------------------------------
with tabs["الحدود والافتراضات"]:
    st.markdown("""
**ما الذي تعرضه هذه الواجهة؟**

| النوع | الحالة |
|---|---|
| بيانات مقاسة | لا توجد في هذا النموذج الأولي |
| بيانات الطقس | اصطناعية (مُحاكاة) ما لم ترفع ملفك الخاص |
| معاملات القاعة | افتراضات هندسية غير مقاسة |
| درجات الحرارة والطاقة | مخرجات محاكاة فيزيائية (نموذج RC) |
| نموذج Random Forest | تنبؤات تعلّم آلي من بيانات المحاكي، اختياري |
| تحدي الكليات | بيانات توضيحية |

**النموذج:** C·dT/dt = UA(T_out − T_in) + Q_solar + Q_internal − Q_cool، بتكامل أويلر الصريح (dt = 15 دقيقة).
Q_solar = A·G·SHGC·(1 − التظليل). Q_internal = N·q + إضاءة + أجهزة (الإضاءة والأجهزة تعمل عند الإشغال فقط).

**الحدود:** نموذج منطقة واحدة بسعة حرارية واحدة؛ تقريب مبسّط لإشعاع الواجهات؛ لا رطوبة ولا تهوية ولا حرارة كامنة؛
قدرة المكيّف وCOP ثابتتان؛ التحكم التنبؤي يفترض توقعاً مثالياً للطقس والإشغال؛ لم يُجرَ أي معايرة أو تحقق ضد قياسات حقيقية؛
**ZILAL لا يتحكم حالياً بأي نظام تكييف فعلي**، والتوفير المعروض تقدير محاكى فقط.
""")
