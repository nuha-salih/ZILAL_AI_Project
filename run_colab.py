"""
run_colab.py — تشغيل ZILAL AI داخل Google Colab فقط (اختبار مؤقت، ليس للنشر).

لماذا تُرفض اتصالات WebSocket في Colab؟
يتحقق Streamlit من ترويسة Origin في اتصال WebSocket مقابل الـ Host. وكيل Colab يعرض الصفحة من
https://8501-....colab.dev بينما يرى الخادم الـ Host الداخلي (…codatalab-user-runtimes.internal)،
فلا يتطابقان فيُرفض الاتصال والصفحة تبقى «تحمّل». تعطيل CORS وحماية XSRF يوقف هذا الفحص.

الأمان: لا يكتب هذا السكربت أي ملف إعداد ولا يغيّر المشروع؛ الإعدادات المخففة تُمرَّر كأعلام
لعملية Streamlit هذه فقط، وتزول بإيقافها. لا تستخدمها في نشر عام.

الاستخدام داخل Colab:
    import run_colab; run_colab.start()
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

PORT = 8501
LOG = "streamlit_colab.log"


def build_command(port: int = PORT) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",            # يوقف فحص Origin/Host للـ WebSocket (Colab فقط)
        "--server.enableXsrfProtection", "false",  # مطلوب مع تعطيل CORS وإلا يُعاد تفعيل CORS
        "--browser.gatherUsageStats", "false",
    ]


def stop() -> None:
    subprocess.run(["pkill", "-f", "streamlit run app.py"], check=False)
    time.sleep(1)


def start(port: int = PORT, height: int = 800, timeout: int = 90, show: bool = True):
    """يوقف أي نسخة سابقة، يشغّل Streamlit، ينتظر جاهزيته، ثم يعرضه في Colab."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    stop()
    log = open(LOG, "w")
    proc = subprocess.Popen(build_command(port), stdout=log, stderr=subprocess.STDOUT)
    url = f"http://localhost:{port}/_stcore/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            break
        try:
            if urllib.request.urlopen(url, timeout=2).read().decode().strip() == "ok":
                print(f"Streamlit جاهز بعد {time.time() - t0:.0f} ث.")
                if show:
                    from google.colab import output  # متاح داخل Colab فقط
                    output.serve_kernel_port_as_iframe(port, height=height)
                    # احتياط: رابط الوكيل المباشر إن لم يظهر الـ iframe (افتحه في تبويب جديد)
                    print("رابط مباشر:", output.eval_js(f"google.colab.kernel.proxyPort({port})"))
                return proc
        except Exception:  # noqa: BLE001
            time.sleep(1)
    print("فشل التشغيل. آخر سطور السجل:\n" + "".join(open(LOG).readlines()[-25:]))
    return proc


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        print(" ".join(build_command()))
    else:
        start(show=False)
