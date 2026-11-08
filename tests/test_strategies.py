"""اختبارات الاستراتيجيات والمقارنة العادلة والمقاييس والتوصيات وتحدي ZILAL الأخضر."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation as ev  # noqa: E402
from forecasting import required_cooling_for_arrival, train_surrogate  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from strategies import StrategyConfig  # noqa: E402
from thermal_model import ClassroomParams, build_sim_inputs, simulate, step_temperature  # noqa: E402
from weather import generate_synthetic_weather, slice_weather  # noqa: E402

DAY = pd.Timestamp("2025-07-14").date()
W = generate_synthetic_weather()


def analysis(key="prelecture", **kw):
    return ev.run_analysis(ClassroomParams(**kw), key, W, DAY, 24)


class TestSavings(unittest.TestCase):
    def test_saving_formula(self):
        self.assertAlmostEqual(ev.saving_percent(100.0, 75.0), 25.0)
        self.assertAlmostEqual(ev.saving_percent(100.0, 120.0), -20.0)
        self.assertEqual(ev.saving_percent(100.0, 100.0), 0.0)

    def test_zero_baseline_safe(self):
        self.assertEqual(ev.saving_percent(0.0, 5.0), 0.0)
        self.assertEqual(ev.saving_percent(float("nan"), 5.0), 0.0)

    def test_summary_savings_consistent(self):
        s = analysis()["summary"]
        base = s.loc["A", "energy_kwh"]
        for k in s.index:
            self.assertAlmostEqual(s.loc[k, "saving_pct"], (base - s.loc[k, "energy_kwh"]) / base * 100)
        self.assertEqual(s.loc["A", "saving_pct"], 0.0)


class TestFairComparison(unittest.TestCase):
    def test_identical_conditions(self):
        r = analysis("solar")
        ref = r["results"]["A"]
        for k, df in r["results"].items():
            for col in ("timestamp", "t_out", "g_facade", "q_solar_w", "q_internal_w", "occupancy"):
                np.testing.assert_array_equal(df[col].to_numpy(), ref[col].to_numpy(), err_msg=f"{k}:{col}")
            self.assertEqual(len(df), len(ref))
            self.assertEqual(df["t_in"].iloc[0], ClassroomParams().initial_temp_c)

    def test_energy_matches_cooling_power(self):
        r = analysis("solar")
        p = ClassroomParams()
        for df in r["results"].values():
            self.assertAlmostEqual(df["e_hvac_kwh"].sum(), (df["q_cool_w"] / p.cop * p.dt_s / 3.6e6).sum())
            self.assertGreaterEqual(df["q_cool_w"].min(), 0.0)
            self.assertLessEqual(df["q_cool_w"].max(), p.hvac_capacity_w + 1e-9)

    def test_trajectory_obeys_rc_equation(self):
        r = analysis("solar")
        inp, p = r["inputs"], ClassroomParams()
        df = r["results"]["D"]
        for k in range(len(df) - 1):
            nxt = step_temperature(df["t_in"][k], inp.t_out[k], inp.q_solar[k], inp.q_internal[k],
                                   df["q_cool_w"][k], p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s)
            self.assertAlmostEqual(nxt, df["t_in"][k + 1], places=9)


class TestStrategiesBehaviour(unittest.TestCase):
    def test_empty_room_fixed_wastes_energy(self):
        s = analysis("empty")["summary"]
        self.assertGreater(s.loc["A", "unnecessary_energy_kwh"], 1.0)
        self.assertLess(s.loc["C", "energy_kwh"], s.loc["A", "energy_kwh"])
        self.assertLess(s.loc["D", "energy_kwh"], s.loc["A", "energy_kwh"])
        self.assertEqual(s.loc["A", "comfort_violation_h"], 0.0)  # لا إشغال → لا مخالفة راحة

    def test_prelecture_predictive_keeps_comfort_rule_based_fails(self):
        s = analysis("prelecture")["summary"]
        self.assertTrue(s.loc["D", "comfort_ok"])
        self.assertFalse(s.loc["C", "comfort_ok"])
        self.assertGreater(s.loc["C", "comfort_violation_h"], s.loc["D", "comfort_violation_h"])
        self.assertLess(s.loc["D", "energy_kwh"], s.loc["A", "energy_kwh"])

    def test_predictive_arrival_temperature_near_target(self):
        r = analysis("prelecture")
        df, inp = r["results"]["D"], r["inputs"]
        j = int(np.argmax(inp.occ > 0))
        self.assertLessEqual(df["t_in"][j], ClassroomParams().comfort_setpoint_c + 0.5 + 0.05)

    def test_required_cooling_formula_exact(self):
        # تطبيق Q المحسوب بثبات يجب أن يوصل T_j إلى الهدف
        r = analysis("prelecture")
        inp, p = r["inputs"], ClassroomParams()
        j = int(np.argmax(inp.occ > 0))
        k, t = j - 12, 29.0
        q = required_cooling_for_arrival(t, k, j, 24.5, inp)
        for i in range(k, j):
            t = step_temperature(t, inp.t_out[i], inp.q_solar[i], inp.q_internal[i], q,
                                 p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s)
        self.assertAlmostEqual(t, 24.5, places=6)

    def test_strategy_config_validation(self):
        for kw in (dict(fixed_on_h=18, fixed_off_h=6), dict(horizon_h=0), dict(deadband_c=-1)):
            with self.assertRaises(ValueError):
                StrategyConfig(**kw).validate()


class TestMetrics(unittest.TestCase):
    def test_comfort_violation_counts_only_occupied_steps(self):
        p = ClassroomParams()
        n = 8
        df = pd.DataFrame({
            "occupancy": [0, 0, 30, 30, 30, 30, 0, 0],
            "t_in": [40.0, 40.0, 24.0, 26.0, 21.0, 24.5, 40.0, 40.0],  # مخالفتان ضمن الإشغال
            "q_cool_w": [0.0] * n, "e_hvac_kwh": [0.0] * n, "q_solar_w": [0.0] * n,
        })

        class FakeInp:
            next_occ = np.array([2, 2, 3, 4, 5, -1, -1, -1])
        m = ev.compute_metrics(df, p, FakeInp())
        self.assertAlmostEqual(m["comfort_violation_h"], 2 * 0.25)  # 26.0 > 25.5 و 21.0 < 22.5
        self.assertAlmostEqual(m["hot_violation_h"], 0.25)
        self.assertAlmostEqual(m["cold_violation_h"], 0.25)
        self.assertAlmostEqual(m["occupied_h"], 1.0)
        self.assertAlmostEqual(m["violation_fraction"], 0.5)
        self.assertFalse(m["comfort_ok"])

    def test_unnecessary_energy_excludes_precondition_window(self):
        p = ClassroomParams()
        occ = np.array([0, 0, 0, 0, 0, 0, 30, 30])
        df = pd.DataFrame({"occupancy": occ, "t_in": [24.0] * 8, "q_cool_w": [0.0] * 8,
                           "e_hvac_kwh": [1.0] * 8, "q_solar_w": [0.0] * 8})

        class FakeInp:
            next_occ = np.array([6, 6, 6, 6, 6, 6, 7, -1])
        m = ev.compute_metrics(df, p, FakeInp(), precond_window_h=0.5)  # نافذة = 2 خطوة
        self.assertAlmostEqual(m["unoccupied_energy_kwh"], 6.0)
        self.assertAlmostEqual(m["unnecessary_energy_kwh"], 4.0)  # الخطوتان 4 و5 ضمن التهيئة

    def test_cop_sensitivity_monotonic(self):
        r = analysis("solar")
        t = ev.cop_sensitivity(r["results"]["C"], ClassroomParams().dt_s)
        self.assertTrue(t["energy_kwh"].is_monotonic_decreasing)


class TestRecommendationsAndSweep(unittest.TestCase):
    def test_shading_sweep_monotonic(self):
        r = analysis("solar")
        sub = r["sweep"][r["sweep"]["orientation"] == "West"].sort_values("shading")
        self.assertTrue(sub["solar_gain_kwh"].is_monotonic_decreasing)
        self.assertTrue(sub["energy_kwh"].is_monotonic_decreasing)

    def test_recommendations_nonempty_and_numeric_backed(self):
        recs = analysis("prelecture")["recommendations"]
        self.assertTrue(recs)
        self.assertTrue(any("kWh" in x for x in recs))

    def test_invalid_scenario_and_params_rejected(self):
        with self.assertRaises(ValueError):
            ev.run_analysis(ClassroomParams(), "nope", W, DAY, 24)
        with self.assertRaises(ValueError):
            ev.run_analysis(ClassroomParams(shading_fraction=2.0), "solar", W, DAY, 24)


class TestGreenChallengeIsolated(unittest.TestCase):
    def test_does_not_affect_core_simulation(self):
        before = analysis("solar")["summary"]["energy_kwh"].to_numpy().copy()
        ev.green_challenge_ranking()
        after = analysis("solar")["summary"]["energy_kwh"].to_numpy()
        np.testing.assert_array_equal(before, after)

    def test_ranking_uses_normalized_improvement_and_comfort(self):
        d = ev.green_challenge_ranking()
        self.assertEqual(list(d["rank"]), [1, 2, 3])
        raw_best = ev.GREEN_CHALLENGE_SAMPLE.sort_values("strategy_kwh").iloc[0]["college"]
        row = d.set_index("college").loc[raw_best]
        self.assertEqual(row["comfort_factor"], 0.0)  # أقل طاقة خام لكن راحة ضعيفة → لا تتصدر
        self.assertNotEqual(d.iloc[0]["college"], raw_best)
        custom = pd.DataFrame({"college": ["x"], "baseline_kwh": [0.0], "strategy_kwh": [3.0], "comfort_compliance": [1.0]})
        self.assertEqual(ev.green_challenge_ranking(custom).iloc[0]["score"], 0.0)


class TestSurrogate(unittest.TestCase):
    def test_surrogate_trains_and_is_labelled(self):
        out = train_surrogate(ClassroomParams(), W, n_samples=800, seed=1)
        self.assertIn("المحاكي", out["note"])
        self.assertLess(out["mae_k_per_step"], out["target_std_k_per_step"])




class TestOvercoolScenario(unittest.TestCase):
    def test_scenario_registered_and_wasteful_baselines(self):
        import evaluation as ev
        from weather import generate_synthetic_weather, start_options
        from thermal_model import ClassroomParams
        self.assertIn("overcool", SCENARIOS)
        w = generate_synthetic_weather()
        start = start_options(w, 24)[0]
        a = ev.run_analysis(ClassroomParams(), "overcool", w, start, 24, __import__("strategies").StrategyConfig())
        sm = a["summary"]
        # A و B يهدران طاقة وقت الخلو أكثر من C و D في المحاضرة القصيرة
        self.assertGreater(sm.loc["A", "unoccupied_energy_kwh"], sm.loc["C", "unoccupied_energy_kwh"])
        self.assertGreater(sm.loc["B", "unoccupied_energy_kwh"], sm.loc["D", "unoccupied_energy_kwh"])


if __name__ == "__main__":
    unittest.main()
