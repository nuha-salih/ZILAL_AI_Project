"""
strategies.py — استراتيجيات التشغيل الأربع. كلها تستخدم الآلية نفسها لتحديد قدرة التبريد
(طلب «يوصل الغرفة إلى الهدف خلال الخطوة» ثم يقيّده المحاكي بقدرة المكيّف)، فتختلف
النتائج بسبب *منطق القرار* فقط:

A — تشغيل ثابت: جدول يومي ثابت (افتراضياً 06:00–18:00) بغض النظر عن الإشغال.
B — حسب الجدول: يعمل أثناء فترات الجدول المعلن فقط (بلا تبريد مسبق).
C — قائم على القواعد: يعمل عند الإشغال وتجاوز الحرارة (هدف + هامش)، وعند خلو القاعة
    لا يبرّد إلا إذا تجاوزت الحرارة حداً وقائياً (لتفادي تراكم الحرارة).
D — تنبؤي: يتنبأ بالحرارة بنموذج RC ويبدأ التبريد المسبق في أبعد لحظة ممكنة تسمح
    بالوصول إلى الهدف عند حضور الطلاب؛ وأثناء الإشغال يستخدم هدفاً أعلى قليلاً داخل نطاق الراحة.
    يفترض توقعاً مثالياً للطقس والإشغال (قيد موثّق).
"""
from __future__ import annotations

from dataclasses import dataclass

from forecasting import required_cooling_for_arrival
from thermal_model import SimInputs, cooling_to_target

STRATEGY_NAMES_AR = {
    "A": "A — تشغيل ثابت",
    "B": "B — حسب الجدول",
    "C": "C — قائم على القواعد",
    "D": "D — تنبؤي",
}


@dataclass
class StrategyConfig:
    fixed_on_h: float = 6.0
    fixed_off_h: float = 18.0
    deadband_c: float = 0.5  # C: التبريد يبدأ عند الهدف + الهامش
    unocc_limit_c: float = 32.0  # حد وقائي عند خلو القاعة (C و D)
    horizon_h: float = 8.0  # أفق التنبؤ لـ D
    predictive_offset_c: float = 0.5  # D: رفع الهدف أثناء الإشغال (داخل نطاق الراحة)
    start_frac: float = 0.85  # D: يبدأ التبريد المسبق عندما يبلغ المطلوب هذه النسبة من القدرة

    def validate(self) -> "StrategyConfig":
        if not (0 <= self.fixed_on_h < self.fixed_off_h <= 24):
            raise ValueError("ساعات التشغيل الثابت غير صالحة (يجب 0 ≤ بداية < نهاية ≤ 24).")
        if not (0 <= self.deadband_c <= 3):
            raise ValueError("هامش القواعد خارج [0, 3].")
        if not (20 <= self.unocc_limit_c <= 40):
            raise ValueError("الحد الوقائي لخلو القاعة خارج [20, 40].")
        if not (0.5 <= self.horizon_h <= 12):
            raise ValueError("أفق التنبؤ خارج [0.5, 12] ساعة.")
        if not (0 <= self.predictive_offset_c <= 2):
            raise ValueError("إزاحة الهدف التنبؤي خارج [0, 2].")
        if not (0.3 <= self.start_frac <= 1.0):
            raise ValueError("نسبة بدء التبريد المسبق خارج [0.3, 1].")
        return self


def _demand(k: int, T: float, inp: SimInputs, target: float) -> float:
    p = inp.params
    return cooling_to_target(T, target, inp.t_out[k], inp.q_solar[k], inp.q_internal[k],
                             p.capacitance_j_per_k, p.ua_w_per_k, p.dt_s)


class _Base:
    key = "?"

    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = (cfg or StrategyConfig()).validate()

    def reset(self, inp: SimInputs) -> None:  # noqa: D401
        pass


class FixedStrategy(_Base):
    key = "A"

    def decide(self, k, T, inp):
        if self.cfg.fixed_on_h <= inp.hours[k] < self.cfg.fixed_off_h:
            return _demand(k, T, inp, inp.params.comfort_setpoint_c)
        return 0.0


class TimetableStrategy(_Base):
    key = "B"

    def decide(self, k, T, inp):
        if inp.timetable_on[k]:
            return _demand(k, T, inp, inp.params.comfort_setpoint_c)
        return 0.0


class RuleBasedStrategy(_Base):
    key = "C"

    def reset(self, inp):
        self.active = False

    def decide(self, k, T, inp):
        sp = inp.params.comfort_setpoint_c
        if inp.occ[k] > 0:
            if T > sp + self.cfg.deadband_c:
                self.active = True
            elif T <= sp + 1e-9:
                self.active = False
            return _demand(k, T, inp, sp) if self.active else 0.0
        self.active = False
        if T > self.cfg.unocc_limit_c:
            return _demand(k, T, inp, self.cfg.unocc_limit_c)
        return 0.0


class PredictiveStrategy(_Base):
    key = "D"

    def reset(self, inp):
        self.precooling = False

    def decide(self, k, T, inp):
        p, cfg = inp.params, self.cfg
        offset = min(cfg.predictive_offset_c, 0.8 * p.comfort_tolerance_c)
        target = p.comfort_setpoint_c + offset
        if inp.occ[k] > 0:
            self.precooling = False
            return _demand(k, T, inp, target)
        horizon_steps = int(cfg.horizon_h * 3600 / p.dt_s)
        j = int(inp.next_occ[k])
        if j != -1 and (j - k) <= horizon_steps:
            q_req = required_cooling_for_arrival(T, k, j, target, inp)
            if q_req >= cfg.start_frac * p.hvac_capacity_w:
                self.precooling = True
            if self.precooling and q_req > 0:
                return q_req
            self.precooling = False
        if T > cfg.unocc_limit_c:
            return _demand(k, T, inp, cfg.unocc_limit_c)
        return 0.0


def make_strategies(cfg: StrategyConfig | None = None) -> dict:
    cfg = (cfg or StrategyConfig()).validate()
    return {"A": FixedStrategy(cfg), "B": TimetableStrategy(cfg),
            "C": RuleBasedStrategy(cfg), "D": PredictiveStrategy(cfg)}
