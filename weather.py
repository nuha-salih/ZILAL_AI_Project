"""
weather.py — بيانات الطقس: توليد مجموعة اصطناعية، وقراءة ملفات CSV مع تحقق.

تنبيه صدق علمي: sample_weather.csv *بيانات محاكاة اصطناعية* لطقس صيفي حار
يشبه المناطق الجنوبية من السعودية؛ وهي ليست قياسات فعلية لنجران.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("timestamp", "outdoor_temp_c", "solar_irradiance_wm2")
ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time"),
    "outdoor_temp_c": ("outdoor_temp_c", "temperature", "temp_c", "t_out", "outdoor_temperature"),
    "solar_irradiance_wm2": ("solar_irradiance_wm2", "ghi", "irradiance", "solar_irradiance"),
}
OPTIONAL_FACADE = ("g_east_wm2", "g_west_wm2", "g_south_wm2")
STEP = pd.Timedelta(minutes=15)


class WeatherDataError(ValueError):
    """خطأ في بيانات الطقس المرفوعة (رسالة عربية واضحة للمستخدم)."""


def generate_synthetic_weather(start: str = "2025-07-14", days: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    يولّد طقساً صيفياً اصطناعياً (خطوة 15 دقيقة):
    • درجة الحرارة: جيب تمام يومي، متوسط 33.5°م وسعة 6.5°م (≈27–40°م) مع اختلاف بين الأيام وضوضاء AR(1).
    • الإشعاع الأفقي GHI: قوس جيبي من 05:45 إلى 18:45 بذروة ≈ 980–1000 W/m².
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 96, freq="15min")
    h = (idx.hour + idx.minute / 60.0).to_numpy()
    day = np.repeat(np.arange(days), 96)
    day_shift = rng.uniform(-1.2, 1.0, size=days)[day]
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.12)
    t = 33.5 + day_shift + 6.5 * np.cos(2 * np.pi * (h - 15.5) / 24.0) + noise
    clear = np.clip(np.sin(np.pi * (h - 5.75) / 13.0), 0, None) ** 1.2
    peak = rng.uniform(950, 1000, size=days)[day]
    cloud = np.clip(1.0 - np.abs(rng.normal(0, 0.02, size=len(idx))), 0.9, 1.0)
    ghi = peak * clear * cloud
    return pd.DataFrame({
        "timestamp": idx,
        "outdoor_temp_c": np.round(t, 2),
        "solar_irradiance_wm2": np.round(ghi, 1),
        "data_source": "SYNTHETIC (simulated, not measured)",
    })


def _find_col(df: pd.DataFrame, canonical: str):
    lower = {c.lower().strip(): c for c in df.columns}
    for a in ALIASES[canonical]:
        if a in lower:
            return lower[a]
    return None


def load_weather_csv(source) -> tuple[pd.DataFrame, list[str]]:
    """
    يقرأ CSV (مسار أو ملف مرفوع) ويتحقق منه. يعيد (DataFrame, تحذيرات).
    يرفع WeatherDataError عند وجود مشكلة تمنع الاستخدام.
    """
    warnings: list[str] = []
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise WeatherDataError(f"تعذّرت قراءة الملف كـ CSV: {exc}") from exc
    if raw.empty:
        raise WeatherDataError("الملف فارغ.")
    cols, missing = {}, []
    for name in REQUIRED:
        c = _find_col(raw, name)
        (missing.append(name) if c is None else cols.__setitem__(name, c))
    if missing:
        raise WeatherDataError(
            "أعمدة مطلوبة مفقودة: " + ", ".join(missing)
            + ". الأعمدة المطلوبة: timestamp, outdoor_temp_c, solar_irradiance_wm2."
        )
    df = pd.DataFrame({k: raw[v] for k, v in cols.items()})
    for extra in OPTIONAL_FACADE:
        if extra in raw.columns:
            df[extra] = raw[extra]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    if before - len(df) > 0:
        warnings.append(f"تم تجاهل {before - len(df)} صفاً بقيم ناقصة/غير صالحة/مكررة.")
    if len(df) < 8:
        raise WeatherDataError("عدد الصفوف الصالحة قليل جداً (يلزم 8 على الأقل).")
    if not df["outdoor_temp_c"].between(-40, 65).all():
        raise WeatherDataError("درجات حرارة خارج المدى المعقول [-40, 65] °م.")
    if not df["solar_irradiance_wm2"].between(0, 1500).all():
        raise WeatherDataError("قيم إشعاع خارج المدى المعقول [0, 1500] W/m².")
    for extra in OPTIONAL_FACADE:
        if extra in df and not df[extra].between(0, 1500).all():
            raise WeatherDataError(f"قيم {extra} خارج المدى [0, 1500].")
    step = df["timestamp"].diff().dropna().median()
    if step > pd.Timedelta(hours=1):
        raise WeatherDataError("الخطوة الزمنية في الملف أكبر من ساعة؛ يلزم بيانات أدق.")
    if step != STEP:
        warnings.append(f"الخطوة الزمنية الأصلية {step}؛ تمت إعادة أخذ العينات إلى 15 دقيقة بالاستيفاء الخطي.")
        df = df.set_index("timestamp").resample("15min").interpolate("time").reset_index()
    df["data_source"] = "UPLOADED"
    return df.reset_index(drop=True), warnings


def start_options(df: pd.DataFrame, hours: int) -> list:
    """التواريخ (منتصف الليل) التي يتوفر بعدها بيانات كافية لمدة المحاكاة."""
    n = int(hours * 4)
    ts = df["timestamp"]
    out = []
    for d in sorted(ts.dt.normalize().unique()):
        pos = ts.index[ts == d]
        if len(pos) and pos[0] + n <= len(df):
            seg = ts.iloc[pos[0]: pos[0] + n]
            if (seg.diff().dropna() == STEP).all():
                out.append(pd.Timestamp(d).date())
    return out


def slice_weather(df: pd.DataFrame, start_date, hours: int) -> pd.DataFrame:
    """يقتطع نافذة المحاكاة بدءاً من منتصف ليل start_date."""
    n = int(hours * 4)
    start = pd.Timestamp(start_date)
    pos = df.index[df["timestamp"] == start]
    if len(pos) == 0 or pos[0] + n > len(df):
        raise WeatherDataError("بيانات الطقس لا تغطي مدة المحاكاة المطلوبة من اليوم المختار.")
    seg = df.iloc[pos[0]: pos[0] + n].reset_index(drop=True)
    if not (seg["timestamp"].diff().dropna() == STEP).all():
        raise WeatherDataError("بيانات الطقس تحتوي على فجوات زمنية داخل نافذة المحاكاة.")
    return seg


if __name__ == "__main__":
    generate_synthetic_weather().to_csv("data/sample_weather.csv", index=False)
    print("wrote data/sample_weather.csv")
