"""Objection categories, their cues, and the suggested broker response for each.

Shared by the rules extractor (cue detection) and the Claude path (fills missing suggestions),
so the broker sees the same playbook answer whichever summarizer ran.
"""

OBJECTION_CUES: dict[str, list[str]] = {
    "price": ["غالي", "غاليه", "السعر عالي", "الاسعار عاليه", "كتير عليا", "فوق ميزانيتي", "expensive", "too high"],
    "down_payment": ["المقدم كبير", "المقدم عالي", "مقدم كبير", "مش معايا المقدم", "down payment is high"],
    "installment": ["القسط كبير", "القسط عالي", "الاقساط عاليه", "القسط مش هقدر", "installment is high"],
    "delivery": ["التسليم بعيد", "متاخر", "اتاخر", "استلام بعد", "التسليم امتى", "delivery is late", "delay"],
    "location": ["بعيد", "المكان بعيد", "مواصلات", "far", "location"],
    "developer": ["المطور مش", "الديفلوبر مش", "سمعه المطور", "مش واثق في المطور", "مش معروف", "developer"],
    "trust": ["خايف", "خائف", "ضمان", "مش ضامن", "نصب", "worried", "guarantee"],
    "timing": ["هفكر", "مش دلوقتي", "بعدين", "لسه بدري", "مش مستعجل", "think about it", "not now"],
    "family_decision": ["اسال مراتي", "اسال جوزي", "اكلم مراتي", "اكلم جوزي", "ارجع لمراتي", "ارجع لجوزي", "my wife", "my husband"],
}

SUGGESTED_RESPONSES: dict[str, str] = {
    "price": "Show 2 units 10-15% below the stated budget in the same area, and a longer installment plan that lowers the monthly amount.",
    "down_payment": "Offer plans with 5-10% down payment or a split down payment over the first 3-6 months.",
    "installment": "Compare the same unit on a longer plan (8-10 years) and show the monthly installment for each.",
    "delivery": "Share ready-to-move or near-delivery units, plus the developer's on-time delivery record.",
    "location": "Send travel times to work/schools and 1-2 alternatives in closer areas.",
    "developer": "Share the developer's delivered projects and resale prices; offer a visit to a delivered compound.",
    "trust": "Explain contract protections and payment to developer accounts; offer a meeting at the office.",
    "timing": "Agree a concrete follow-up date and send a short shortlist so the decision has something to anchor on.",
    "family_decision": "Invite the spouse to a joint call or site visit; send a one-page comparison they can review together.",
    "other": "Acknowledge and ask one clarifying question before proposing options.",
}

BROKER_COMMITMENT_CUES = ["هبعتلك", "هبعت لحضرتك", "هكلمك", "هكلم حضرتك", "هرتب", "هحجزلك", "هحجز", "هتابع", "هرجعلك",
                          "i'll send", "i will send", "i'll call", "will call you back"]
CLIENT_COMMITMENT_CUES = ["هفكر وارد", "هرد عليك", "هرجعلك", "هكلمك", "هنزل معاينه", "هاجي", "هبعتلك", "هحول",
                          "i'll get back", "i will visit"]

POSITIVE_CUES = ["تمام", "ممتاز", "حلو", "عاجبني", "عجبني", "جميل", "مناسب", "موافق", "يلا", "great", "perfect"]
NEGATIVE_CUES = ["غالي", "مش مناسب", "مش عاجبني", "لا مش", "صعب", "مش هينفع", "زعلان", "not interested", "too expensive"]

PAYMENT_CUES = {
    "mortgage": ["تمويل عقاري", "مورتجدج", "مورجيج", "قرض", "البنك", "mortgage", "ناوي ناو", "nawy now"],
    "installments": ["تقسيط", "قسط", "اقساط", "installment", "installments"],
    "cash": ["كاش", " كش", "cash", "دفعه واحده"],  # " كش": ASR spelling; leading space so "كشف" does not match
}
