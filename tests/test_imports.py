"""اختبار استيراد كل وحدات المشروع (app.py يُستورد فقط إن توفر streamlit/plotly)."""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    def test_core_modules_import(self):
        for m in ("thermal_model", "weather", "scenarios", "forecasting", "strategies", "evaluation", "check_env", "run_colab"):
            importlib.import_module(m)

    def test_colab_command_flags(self):
        import run_colab
        cmd = " ".join(run_colab.build_command())
        self.assertIn("--server.enableCORS false", cmd)
        self.assertIn("--server.enableXsrfProtection false", cmd)
        self.assertIn("--server.headless true", cmd)

    def test_no_insecure_config_file_left(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertFalse(os.path.exists(os.path.join(root, ".streamlit", "config.toml")))

    def test_check_env_runs(self):
        import check_env
        self.assertIn(check_env.try_import("numpy")[0], (True,))
        self.assertFalse(check_env.try_import("definitely_missing_pkg")[0])


if __name__ == "__main__":
    unittest.main()
