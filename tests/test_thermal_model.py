"""اختبارات الحسابات الحرارية الأساسية (unittest — تعمل أيضاً مع pytest)."""
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermal_model import (ClassroomParams, build_sim_inputs, clip_cooling, cooling_to_target,  # noqa: E402
                           facade_irradiance, hvac_energy_kwh, internal_gain, simulate,
                           solar_gain, step_temperature)
from weather import (WeatherDataError, generate_synthetic_weather, load_weather_csv,  # noqa: E402
                     slice_weather)


class Const:
    """استراتيجية اختبار: قدرة تبريد ثابتة مطلوبة."""
    def __init__(self, q): self.q = q
    def reset(self, inp): pass
    def decide(self, k, T, inp): return self.q


def weather_day():
    return slice_weather(generate_synthetic_weather(), pd.Timestamp("2025-07-14").date(), 24)


class TestPhysics(unittest.TestCase):
    def test_euler_step_matches_equation(self):
        t = step_temperature(24.0, 40.0, 1000.0, 2000.0, 3000.0, c=9e6, ua=200.0, dt=900.0)
        expected = 24.0 + 900.0 / 9e6 * (200.0 * 16.0 + 1000.0 + 2000.0 - 3000.0)
        self.assertAlmostEqual(t, expected, places=12)

    def test_solar_gain_formula_and_shading(self):
        g = np.array([0.0, 500.0, 800.0])
        q = solar_gain(12.0, g, 0.6, 0.0)
        np.testing.assert_allclose(q, 12.0 * g * 0.6)
        self.assertTrue(np.all(solar_gain(12.0, g, 0.6, 0.5) <= q))
        np.testing.assert_allclose(solar_gain(12.0, g, 0.6, 1.0), 0.0)

    def test_increasing_shading_reduces_solar_gain(self):
        w = weather_day()
        tot = []
        for s in (0.0, 0.3, 0.6, 0.9):
            inp = build_sim_inputs(ClassroomParams(shading_fraction=s), w, [(8, 16)], [(8, 16)])
            tot.append(inp.q_solar.sum())
        self.assertTrue(all(a > b for a, b in zip(tot, tot[1:])))

    def test_occupancy_changes_internal_gain(self):
        p = ClassroomParams()
        self.assertGreater(internal_gain(p, 30), internal_gain(p, 10))
        self.assertAlmostEqual(internal_gain(p, 30) - internal_gain(p, 10), 20 * p.q_person_w)
        self.assertEqual(internal_gain(p, 0), 0.0)  # إضاءة/أجهزة مطفأة عند الخلو
        self.assertAlmostEqual(internal_gain(p, 30), 30 * 75 + 12 * 60 + 300)

    def test_orientation_changes_exposure_profile(self):
        w = weather_day()
        peaks = {}
        for o in ("East", "West", "South"):
            inp = build_sim_inputs(ClassroomParams(orientation=o), w, [], [])
            peaks[o] = inp.hours[int(np.argmax(inp.q_solar))]
        self.assertLess(peaks["East"], 11.0)
        self.assertGreater(peaks["West"], 13.0)
        self.assertTrue(11.0 <= peaks["South"] <= 13.5)
        tot = {o: build_sim_inputs(ClassroomParams(orientation=o), w, [], []).q_solar.sum()
               for o in ("East", "West", "South")}
        self.assertLess(tot["South"], tot["East"])
        self.assertLess(tot["South"], tot["West"])

    def test_facade_irradiance_zero_at_night(self):
        h = np.array([0.0, 3.0, 22.0])
        np.testing.assert_allclose(facade_irradiance("West", h, np.zeros(3)), 0.0)

    def test_cooling_clipped_nonnegative_and_capped(self):
        self.assertEqual(clip_cooling(-500.0, 15000.0), 0.0)
        self.assertEqual(clip_cooling(99999.0, 15000.0), 15000.0)
        inp = build_sim_inputs(ClassroomParams(hvac_capacity_w=8000.0), weather_day(), [(8, 16)], [(8, 16)])
        df = simulate(inp, Const(1e9))
        self.assertLessEqual(df["q_cool_w"].max(), 8000.0 + 1e-9)
        df0 = simulate(inp, Const(-1e9))
        self.assertGreaterEqual(df0["q_cool_w"].min(), 0.0)

    def test_more_cooling_gives_lower_indoor_temperature(self):
        w = weather_day()
        end = []
        for cap in (4000.0, 8000.0, 15000.0):
            inp = build_sim_inputs(ClassroomParams(hvac_capacity_w=cap), w, [(8, 16)], [(8, 16)])
            end.append(simulate(inp, Const(1e9))["t_in"].mean())
        self.assertTrue(end[0] > end[1] > end[2])

    def test_higher_cop_reduces_energy_same_cooling(self):
        q = np.full(96, 5000.0)
        self.assertLess(hvac_energy_kwh(q, 3.5).sum(), hvac_energy_kwh(q, 2.5).sum())
        self.assertAlmostEqual(hvac_energy_kwh(np.array([3000.0]), 3.0, 900.0)[0], 3000 / 3 * 900 / 3.6e6)
        w = weather_day()
        e = [simulate(build_sim_inputs(ClassroomParams(cop=c), w, [(8, 16)], [(8, 16)]), Const(6000.0))["e_hvac_kwh"].sum()
             for c in (2.5, 3.0, 3.5)]
        self.assertTrue(e[0] > e[1] > e[2])

    def test_deadbeat_target_reaches_setpoint(self):
        p = ClassroomParams()
        q = cooling_to_target(30.0, 24.0, 38.0, 2000.0, 3000.0, p.capacitance_j_per_k, p.ua_w_per_k)
        t = step_temperature(30.0, 38.0, 2000.0, 3000.0, q, p.capacitance_j_per_k, p.ua_w_per_k)
        self.assertAlmostEqual(t, 24.0, places=9)

    def test_free_float_energy_conservation_steady_state(self):
        # بلا مكاسب وبدرجة خارجية ثابتة تتقارب T_in إلى T_out ولا تتجاوزها
        t = 24.0
        p = ClassroomParams()
        for _ in range(4000):
            t = step_temperature(t, 35.0, 0.0, 0.0, 0.0, p.capacitance_j_per_k, p.ua_w_per_k)
        self.assertAlmostEqual(t, 35.0, places=2)
        self.assertLessEqual(t, 35.0 + 1e-9)


class TestValidation(unittest.TestCase):
    def test_invalid_inputs_rejected(self):
        bad = [dict(area_m2=-5), dict(occupancy=-1), dict(occupancy=31), dict(shading_fraction=1.5),
               dict(shading_fraction=-0.1), dict(orientation="North"), dict(cop=0),
               dict(hvac_capacity_w=0), dict(shgc=2.0), dict(area_m2=float("nan")),
               dict(ua_w_per_k_m2=200.0)]  # غير مستقر عددياً
        for kw in bad:
            with self.assertRaises(ValueError, msg=str(kw)):
                ClassroomParams(**kw).validate()

    def test_defaults_valid(self):
        ClassroomParams().validate()


class TestWeather(unittest.TestCase):
    def test_sample_csv_loads(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df, warns = load_weather_csv(os.path.join(root, "data", "sample_weather.csv"))
        self.assertEqual(len(df), 384)
        self.assertEqual(warns, [])

    def test_missing_columns_graceful(self):
        with self.assertRaises(WeatherDataError):
            load_weather_csv(io.StringIO("timestamp,foo\n2025-07-14 00:00,3\n"))

    def test_invalid_values_graceful(self):
        rows = "\n".join(f"2025-07-14 {h:02d}:00,{t},{g}" for h, t, g in
                         [(0, 30, 0), (1, 29, 0), (2, 29, 0), (3, 28, 0), (4, 28, 0), (5, 28, 0), (6, 29, 5), (7, 100, 100)])
        with self.assertRaises(WeatherDataError):
            load_weather_csv(io.StringIO("timestamp,outdoor_temp_c,solar_irradiance_wm2\n" + rows))
        with self.assertRaises(WeatherDataError):
            load_weather_csv(io.StringIO(""))
        with self.assertRaises(WeatherDataError):
            load_weather_csv(io.StringIO("timestamp,outdoor_temp_c,solar_irradiance_wm2\nbad,x,y\n"))

    def test_hourly_data_resampled(self):
        rows = "\n".join(f"2025-07-14 {h:02d}:00,{30 + h * 0.1},{max(0, (h - 6) * 50)}" for h in range(24))
        df, warns = load_weather_csv(io.StringIO("timestamp,outdoor_temp_c,solar_irradiance_wm2\n" + rows))
        self.assertTrue(warns)
        self.assertEqual(len(df), 23 * 4 + 1)

    def test_slice_requires_coverage(self):
        df = generate_synthetic_weather(days=1)
        with self.assertRaises(WeatherDataError):
            slice_weather(df, pd.Timestamp("2025-07-14").date(), 48)


if __name__ == "__main__":
    unittest.main()
