"""Evaluation metrics: ASR (WER/CER), diarization (DER), CRM field accuracy."""

from app.text.normalize import normalize_search


def wer_cer(reference: str, hypothesis: str) -> dict[str, float]:
    """WER/CER on search-normalized text, so alef/ya/teh-marbuta spelling variants are not counted as errors."""
    import jiwer

    ref, hyp = normalize_search(reference), normalize_search(hypothesis)
    return {"wer": round(jiwer.wer(ref, hyp), 4), "cer": round(jiwer.cer(ref, hyp), 4)}


Turn = tuple[float, float, str]


def der(reference: list[Turn], hypothesis: list[Turn], step: float = 0.01, collar: float = 0.25) -> float:
    """Frame-level diarization error rate with optimal 2-speaker label mapping and a forgiveness collar
    around reference boundaries (same convention as pyannote.metrics' default evaluation).
    Uses pyannote.metrics when installed."""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        def ann(turns: list[Turn]) -> Annotation:
            a = Annotation()
            for s, e, label in turns:
                a[Segment(s, e)] = label
            return a

        return round(float(DiarizationErrorRate(collar=collar * 2)(ann(reference), ann(hypothesis))), 4)
    except ImportError:
        pass
    from itertools import permutations

    end = max([e for _, e, _ in reference + hypothesis] or [0])
    n = int(end / step) + 1

    def frames(turns: list[Turn]) -> list[str | None]:
        out: list[str | None] = [None] * n
        for s, e, label in turns:
            for i in range(int(s / step), min(n, int(e / step))):
                out[i] = label
        return out

    ref, hyp = frames(reference), frames(hypothesis)
    skip = set()
    for s, e, _ in reference:
        for t in (s, e):
            skip.update(range(max(0, int((t - collar) / step)), min(n, int((t + collar) / step))))
    ref_labels = sorted({r for r in ref if r})
    hyp_labels = sorted({h for h in hyp if h})
    best = None
    for perm in permutations(ref_labels, min(len(ref_labels), len(hyp_labels))):
        mapping = dict(zip(hyp_labels, perm, strict=False))
        errors = total = 0
        for i in range(n):
            if i in skip or ref[i] is None and hyp[i] is None:
                continue
            if ref[i] is not None:
                total += 1
            mapped = mapping.get(hyp[i]) if hyp[i] else None
            if ref[i] != mapped:
                errors += 1
        rate = errors / total if total else 0.0
        best = rate if best is None else min(best, rate)
    return round(best or 0.0, 4)


def _norm(v):
    if isinstance(v, list):
        return sorted(str(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def field_scores(gold: dict, predicted: dict) -> dict[str, bool]:
    """Exact match per gold field (lists compared as sets; money within 2%)."""
    out = {}
    for field, expected in gold.items():
        got = predicted.get(field)
        if isinstance(expected, int | float) and isinstance(got, int | float) and expected:
            out[field] = abs(got - expected) / expected <= 0.02
        else:
            out[field] = _norm(expected) == _norm(got)
    return out
