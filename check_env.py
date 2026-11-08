"""
check_env.py — يفحص البيئة قبل التشغيل ويشخّص سبب فشل الاستيراد (يعمل على Windows وLinux وColab).
    python check_env.py
رمز الخروج 0 = كل الحزم الأساسية تعمل، 1 = هناك مشكلة.
"""
from __future__ import annotations

import importlib
import platform
import sys

CORE = ["numpy", "pandas", "plotly", "streamlit"]
OPTIONAL = ["sklearn"]


def try_import(name: str):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?"), ""
    except BaseException as exc:  # noqa: BLE001  (DLL load failures are ImportError, but be broad)
        return False, "", f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"Python {sys.version.split()[0]} ({platform.architecture()[0]}) on {platform.system()} — {sys.executable}")
    if sys.version_info >= (3, 14):
        print("ملاحظة: Python 3.14 حديث جداً؛ بعض العجلات المترجمة قد لا تكون مألوفة لسياسات الأمان. المختبَر هنا: 3.12.")
    bad = 0
    for name in CORE + OPTIONAL:
        ok, ver, err = try_import(name)
        tag = "OK     " if ok else ("MISSING" if name in OPTIONAL else "FAILED ")
        print(f"[{tag}] {name} {ver} {err}")
        if not ok and name in CORE:
            bad += 1
            if "application control" in err.lower() or "dll load failed" in err.lower():
                print("   ↳ سياسة Windows (Smart App Control / WDAC / AppLocker) تمنع ملفاً مترجماً (.pyd/.dll) داخل الحزمة؛ "
                      "هذه ليست مشكلة في كود ZILAL. جرّب نسخة/بناء آخر للحزمة (setup_windows.ps1 يفعل ذلك تلقائياً)،"
                      " أو اطلب من IT السماح بالملف، أو شغّل المشروع على Colab/WSL.")
    if bad == 0:
        for mod in ["thermal_model", "weather", "scenarios", "forecasting", "strategies", "evaluation"]:
            ok, _, err = try_import(mod)
            print(f"[{'OK     ' if ok else 'FAILED '}] project module {mod} {err}")
            bad += 0 if ok else 1
    print("النتيجة:", "جاهز ✔" if bad == 0 else "توجد مشاكل ✘")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
