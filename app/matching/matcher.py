"""Unit matcher: hybrid retrieval over inventory, then constraint scoring.

1. Lexical: BM25 over normalized unit text (Arabic-folded, so area/type spellings match).
2. Dense: sentence-transformers (bge-m3) when installed, else char n-gram TF-IDF cosine.
   In production the dense vectors live in pgvector; the in-memory index here holds the same
   vectors for a demo-sized inventory.
3. Reciprocal rank fusion of both lists, then a constraint score from the structured needs
   (budget fit, area, type, bedrooms, down payment, plan length, delivery). Hard filters: unit type,
   bedrooms within 1, status available, price within 115% of the budget ceiling.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import get_settings
from app.domain import Needs, UnitMatch
from app.text.gazetteer import AREAS, UNIT_TYPES
from app.text.normalize import normalize_search
from app.text.numbers import format_egp

log = logging.getLogger(__name__)

RRF_K = 60
BUDGET_STRETCH = 1.15


def unit_text(u: dict) -> str:
    """English + Arabic aliases so a query in either language hits the unit."""
    area_aliases = " ".join(AREAS.get(u["area"], [])[:3])
    type_aliases = " ".join(UNIT_TYPES.get(u["unit_type"], [])[:2])
    return normalize_search(
        f"{u['title']} {u['compound']} {u['developer']} {u['area']} {area_aliases} {u['unit_type']} "
        f"{type_aliases} {u['bedrooms']} bedrooms {u['finishing']} delivery {u['delivery']} "
        f"{'near schools' if u.get('near_schools') else ''}"
    )


class Matcher:
    def __init__(self, units: list[dict]):
        self.units = units
        texts = [unit_text(u) for u in units]
        self.bm25 = BM25Okapi([t.split() for t in texts])
        self.encoder = None
        dense_model = get_settings().dense_model
        if dense_model:
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(dense_model)
                self.vectors = self.encoder.encode(texts, normalize_embeddings=True)
            except Exception:
                log.exception("dense model %s unavailable; using TF-IDF", dense_model)
                self.encoder = None
        if self.encoder is None:  # TF-IDF char n-grams keep the dense leg working without a model
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self.vectors = self.tfidf.fit_transform(texts)

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.encoder is not None:
            q = self.encoder.encode([query], normalize_embeddings=True)
            return (self.vectors @ q.T).ravel()
        q = self.tfidf.transform([query])
        return (self.vectors @ q.T).toarray().ravel()

    def retrieve(self, query: str, k: int = 50) -> dict[int, float]:
        """RRF-fused rank scores for the top-k of each retriever."""
        q = normalize_search(query)
        fused: dict[int, float] = {}
        for scores in (self.bm25.get_scores(q.split()), self._dense_scores(q)):
            for rank, idx in enumerate(np.argsort(-scores)[:k]):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        return fused

    def match(self, needs: Needs, query_text: str = "", top_k: int = 3) -> list[UnitMatch]:
        query = " ".join(filter(None, [
            " ".join(needs.areas), str(needs.unit_type.value or ""),
            f"{needs.bedrooms.value} bedrooms" if needs.bedrooms.value else "",
            "ready" if needs.delivery.value == "ready" else "", query_text,
        ])) or "apartment"
        fused = self.retrieve(query)
        max_rrf = max(fused.values()) if fused else 1.0
        budget = needs.budget_max_egp.value
        ranked: list[UnitMatch] = []
        for idx, unit in enumerate(self.units):
            if unit.get("status") != "available":
                continue
            if budget and unit["price_egp"] > float(budget) * BUDGET_STRETCH:
                continue
            if not self._fits_client(unit, needs):
                continue
            constraint, reasons = self._constraint_score(unit, needs)
            retrieval = fused.get(idx, 0.0) / max_rrf
            score = 0.7 * constraint + 0.3 * retrieval
            ranked.append(UnitMatch(
                unit_id=unit["unit_id"], title=unit["title"], compound=unit["compound"], area=unit["area"],
                unit_type=unit["unit_type"], bedrooms=unit["bedrooms"], price_egp=unit["price_egp"],
                down_payment_pct=unit["down_payment_pct"], installment_years=unit["installment_years"],
                delivery=unit["delivery"], score=round(score, 3), reasons=reasons,
            ))
        ranked.sort(key=lambda m: m.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _fits_client(u: dict, n: Needs) -> bool:
        """Hard filter: a unit of the wrong type, or 2+ bedrooms off, is no use to the client however well
        it fits area and budget (a 1BR studio for a family asking for 3BR). Area stays soft."""
        if n.unit_type.value and u["unit_type"] != n.unit_type.value:
            return False
        return not (n.bedrooms.value and abs(u["bedrooms"] - int(n.bedrooms.value)) >= 2)

    @staticmethod
    def _constraint_score(u: dict, n: Needs) -> tuple[float, list[str]]:
        score, weight, reasons = 0.0, 0.0, []

        def add(w: float, ok: float, reason: str | None) -> None:
            nonlocal score, weight
            weight += w
            score += w * ok
            if reason and ok >= 0.99:
                reasons.append(reason)

        if n.areas:
            add(3, 1.0 if u["area"] in n.areas else 0.0, f"in {u['area']}")
        if n.unit_type.value:
            add(2, 1.0 if u["unit_type"] == n.unit_type.value else 0.0, f"{u['unit_type']} as requested")
        if n.bedrooms.value:
            diff = abs(u["bedrooms"] - int(n.bedrooms.value))
            add(1.5, 1.0 if diff == 0 else 0.4 if diff == 1 else 0.0, f"{u['bedrooms']} bedrooms")
        budget = n.budget_max_egp.value
        if budget:
            ratio = u["price_egp"] / float(budget)
            ok = 1.0 if 0.75 <= ratio <= 1.0 else 0.6 if ratio < 0.75 else max(0.0, 1 - (ratio - 1) / 0.15)
            add(3, ok, f"{format_egp(u['price_egp'])} within budget {format_egp(budget)}")
        if n.down_payment_egp.value:
            needed = u["price_egp"] * u["down_payment_pct"] / 100
            add(1.5, 1.0 if needed <= float(n.down_payment_egp.value) * 1.05 else 0.3,
                f"down payment {format_egp(needed)} fits")
        if n.installment_years.value:
            add(1, 1.0 if u["installment_years"] >= int(n.installment_years.value) else 0.3,
                f"{u['installment_years']}-year plan")
        if n.payment_plan.value == "installments":
            add(0.5, 1.0 if u["installment_years"] > 0 else 0.0, None)
        if n.delivery.value == "ready":
            add(1.5, 1.0 if u["delivery"] == "ready" else 0.0, "ready to move")
        if weight == 0:
            return 0.5, ["no stated constraints yet"]
        return score / weight, reasons


def load_inventory(path: str | None = None) -> list[dict]:
    p = Path(path or get_settings().inventory_path)
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache
def get_matcher() -> Matcher:
    return Matcher(load_inventory())
