"""Text inputs -> Transcript: WhatsApp chat exports and speaker-prefixed call notes/transcripts.

WhatsApp export lines (Android and iOS):
  12/03/2026, 14:05 - Ahmed Nawy: مساء الخير
  [12/03/2026, 2:05:33 PM] Client Name: عايز شقة في التجمع
Multi-line messages continue on lines without a timestamp prefix.
"""

import re
from datetime import datetime

from app.domain import Segment, Speaker, Transcript
from app.text.normalize import clean_display

WA_LINE = re.compile(
    r"^\[?(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),?\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*[-–]?\s*"
    r"(?P<who>[^:]{1,60}):\s(?P<msg>.*)$"
)
SPEAKER_LINE = re.compile(
    r"^\s*(?:\[(?P<ts>\d{1,2}:\d{2})\]\s*)?(?P<who>broker|client|agent|customer|بروكر|عميل|العميل|السيلز)\s*[:：]\s*(?P<msg>.+)$",
    re.IGNORECASE,
)
BROKER_NAMES = {"broker", "agent", "بروكر", "السيلز"}
SYSTEM_MESSAGES = ("<Media omitted>", "<الوسائط غير مضمنة>", "Messages and calls are end-to-end encrypted",
                   "This message was deleted", "تم حذف هذه الرسالة")


def _parse_dt(date: str, time: str) -> datetime | None:
    time = time.upper().replace(" ", "")
    for fmt in ("%d/%m/%Y%H:%M", "%d/%m/%y%H:%M", "%d/%m/%Y%H:%M:%S", "%d/%m/%Y%I:%M%p",
                "%d/%m/%Y%I:%M:%S%p", "%d/%m/%y%I:%M%p", "%m/%d/%y%I:%M%p"):
        try:
            return datetime.strptime(date + time, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_export(text: str, broker_names: list[str]) -> Transcript:
    """Each message becomes one segment; timestamps are seconds since the first message."""
    broker_keys = {n.strip().lower() for n in broker_names if n.strip()}
    rows: list[tuple[datetime | None, str, str]] = []
    for line in text.splitlines():
        m = WA_LINE.match(line.strip("‎‏ "))
        if m:
            rows.append((_parse_dt(m["date"], m["time"]), m["who"].strip(), m["msg"]))
        elif rows and line.strip():
            dt, who, msg = rows[-1]
            rows[-1] = (dt, who, f"{msg}\n{line}")
    rows = [r for r in rows if not any(sm in r[2] for sm in SYSTEM_MESSAGES)]
    if not rows:
        return Transcript()
    t0 = next((r[0] for r in rows if r[0]), None)
    segments = []
    for i, (dt, who, msg) in enumerate(rows):
        start = (dt - t0).total_seconds() if dt and t0 else float(i)
        is_broker = who.lower() in broker_keys or any(k in who.lower() for k in broker_keys)
        segments.append(Segment(
            start=start, end=start, text=clean_display(msg), speaker_raw=who,
            speaker=Speaker.BROKER if is_broker else Speaker.CLIENT,
        ))
    return Transcript(duration_s=segments[-1].start, segments=segments, asr_model="whatsapp-text")


def parse_speaker_text(text: str) -> Transcript:
    """'Broker: ...' / 'Client: ...' lines (manual notes, synthetic eval calls, external transcripts)."""
    segments: list[Segment] = []
    for i, line in enumerate(text.splitlines()):
        m = SPEAKER_LINE.match(line)
        if not m:
            if segments and line.strip():
                segments[-1].text += " " + clean_display(line)
            continue
        who = m["who"].lower()
        if m["ts"]:
            mm, ss = m["ts"].split(":")
            start = int(mm) * 60 + int(ss)
        else:
            start = float(i * 5)
        segments.append(Segment(
            start=start, end=start + 5, text=clean_display(m["msg"]), speaker_raw=who,
            speaker=Speaker.BROKER if who in BROKER_NAMES else Speaker.CLIENT,
        ))
    duration = segments[-1].end if segments else 0.0
    return Transcript(duration_s=duration, segments=segments, asr_model="text")


def looks_like_whatsapp(text: str) -> bool:
    lines = [ln for ln in text.splitlines()[:10] if ln.strip()]
    return sum(bool(WA_LINE.match(ln.strip("‎‏ "))) for ln in lines) >= max(1, len(lines) // 2)
