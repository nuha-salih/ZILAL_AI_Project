"""scenarios.py — السيناريوهات التجريبية الأربعة (جداول بالساعات 0–24)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    name_ar: str
    description_ar: str
    occupancy_blocks: tuple  # الإشغال الفعلي
    timetable_blocks: tuple  # الجدول المعلن (تعتمد عليه الاستراتيجية B)
    duration_h: int = 24


SCENARIOS = {
    "empty": Scenario(
        "empty", "١) قاعة فارغة",
        "القاعة محجوزة في الجدول من 08:00 إلى 14:00 لكن لا يحضر أحد. يبيّن هذا التشغيل غير الضروري "
        "عندما يعتمد التحكم على جدول ثابت أو ساعات دوام بدل الإشغال الفعلي.",
        occupancy_blocks=(), timetable_blocks=((8.0, 14.0),),
    ),
    "prelecture": Scenario(
        "prelecture", "٢) ما قبل المحاضرة",
        "محاضرة واحدة من 12:00 إلى 14:00 في وقت الظهيرة الحار. يبيّن كيف يمكن تهيئة القاعة قبل وصول "
        "الطلاب بدل بدء التبريد عند الحضور فقط.",
        occupancy_blocks=((12.0, 14.0),), timetable_blocks=((12.0, 14.0),),
    ),
    "solar": Scenario(
        "solar", "٣) كسب شمسي مرتفع",
        "محاضرات من 08:00 إلى 16:00 مع تعرّض الواجهة للشمس. غيّر اتجاه النافذة ونسبة التظليل لترى أثرهما "
        "على الكسب الشمسي وطلب التبريد.",
        occupancy_blocks=((8.0, 16.0),), timetable_blocks=((8.0, 16.0),),
    ),
    "overcool": Scenario(
        "overcool", "٤) إفراط في التبريد (محاضرة قصيرة)",
        "الجدول المعلن 08:00–16:00 لكن المحاضرة الفعلية الوحيدة من 10:00 إلى 11:30 فقط. التشغيل الثابت (A) وحسب "
        "الجدول (B) يواصلان تبريد قاعة فارغة ساعات بعد الانتهاء؛ انظر «طاقة وقت الخلو» و«طاقة غير ضرورية».",
        occupancy_blocks=((10.0, 11.5),), timetable_blocks=((8.0, 16.0),),
    ),
}
