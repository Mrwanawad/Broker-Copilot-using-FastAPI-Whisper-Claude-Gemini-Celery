"""Generate the synthetic inventory and CRM leads used by the demo and tests (seeded, reproducible).

Prices are plausible 2026 EGP ranges per area/type, not real listings.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPOUNDS = {
    "New Cairo": [("Palm Hills New Cairo", "Palm Hills"), ("Mountain View iCity", "Mountain View"),
                  ("Hyde Park", "Hyde Park"), ("Villette", "SODIC")],
    "Sheikh Zayed": [("Zed West", "Ora"), ("Karmell", "Sodic"), ("Belle Vie", "Emaar Misr")],
    "6th of October": [("O West", "Orascom"), ("Badya", "Palm Hills"), ("Sodic East Town West", "SODIC")],
    "New Capital": [("Il Bosco City", "Misr Italia"), ("Midtown Condo", "Better Home")],
    "North Coast": [("Marassi", "Emaar Misr"), ("Hacienda West", "Palm Hills"), ("Silver Sands", "Ora")],
    "Mostakbal City": [("Bloomfields", "Tatweer Misr"), ("Sarai", "MNHD")],
    "Ain Sokhna": [("Il Monte Galala", "Tatweer Misr")],
}
# (type, bedrooms options, price per sqm EGP range, size range sqm)
TYPES = {
    "apartment": ([1, 2, 3], (55_000, 85_000), (70, 190)),
    "duplex": ([3, 4], (60_000, 90_000), (180, 280)),
    "townhouse": ([3, 4], (70_000, 100_000), (190, 260)),
    "twin house": ([4], (75_000, 110_000), (240, 320)),
    "villa": ([4, 5], (80_000, 130_000), (300, 500)),
    "chalet": ([1, 2, 3], (70_000, 120_000), (60, 150)),
    "studio": ([1], (55_000, 80_000), (40, 65)),
}
AREA_TYPES = {
    "North Coast": ["chalet", "townhouse", "twin house", "villa"],
    "Ain Sokhna": ["chalet", "townhouse"],
}
DELIVERY = ["ready", "2027", "2028", "2029"]
FINISHING = ["fully finished", "semi finished", "core and shell"]


def make_units(n: int = 120, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    units = []
    for i in range(n):
        area = rng.choice(list(COMPOUNDS))
        compound, developer = rng.choice(COMPOUNDS[area])
        unit_type = rng.choice(AREA_TYPES.get(area, ["apartment", "apartment", "apartment", "duplex",
                                                     "townhouse", "twin house", "villa", "studio"]))
        beds_opts, ppsqm, size_rng = TYPES[unit_type]
        size = rng.randint(*size_rng)
        price = round(size * rng.randint(*ppsqm), -4)
        delivery = rng.choice(DELIVERY)
        years = 0 if delivery == "ready" and rng.random() < 0.3 else rng.choice([5, 6, 7, 8, 10])
        down = 100.0 if years == 0 else rng.choice([5, 10, 10, 15, 20])
        units.append({
            "unit_id": f"U{1000 + i}",
            "title": f"{beds_opts[-1] if unit_type == 'studio' else rng.choice(beds_opts)}BR {unit_type} in {compound}",
            "compound": compound, "developer": developer, "area": area, "unit_type": unit_type,
            "bedrooms": 1 if unit_type == "studio" else rng.choice(beds_opts),
            "size_sqm": size, "price_egp": int(price), "down_payment_pct": float(down),
            "installment_years": years, "delivery": delivery, "finishing": rng.choice(FINISHING),
            "near_schools": rng.random() < 0.5, "status": "available" if rng.random() < 0.9 else "reserved",
        })
    for u in units:  # title must agree with the bedrooms field
        u["title"] = f"{u['bedrooms']}BR {u['unit_type']} in {u['compound']}"
    return units


LEADS = [
    {"lead_id": "L-1001", "name": "Ahmed Samir", "phone": "+201000000001", "owner": "broker.mona",
     "stage": "new", "preferred_areas": [], "unit_type": None, "bedrooms": None, "budget_min": None,
     "budget_max": None, "down_payment": None, "payment_method": None, "installment_years": None,
     "delivery_preference": None, "purpose": None, "next_follow_up": None, "source": "Facebook ad"},
    {"lead_id": "L-1002", "name": "Sara Hassan", "phone": "+201000000002", "owner": "broker.mona",
     "stage": "qualified", "preferred_areas": ["Sheikh Zayed"], "unit_type": "apartment", "bedrooms": 3,
     "budget_min": None, "budget_max": 6_000_000, "down_payment": None, "payment_method": "installments",
     "installment_years": None, "delivery_preference": None, "purpose": "living", "next_follow_up": None,
     "source": "Website"},
    {"lead_id": "L-1003", "name": "Karim Adel", "phone": "+201000000003", "owner": "broker.omar",
     "stage": "viewing", "preferred_areas": ["North Coast"], "unit_type": "chalet", "bedrooms": 2,
     "budget_min": None, "budget_max": 9_000_000, "down_payment": None, "payment_method": None,
     "installment_years": None, "delivery_preference": None, "purpose": "summer", "next_follow_up": None,
     "source": "Referral"},
]


def main() -> None:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / "inventory.json").write_text(json.dumps(make_units(), ensure_ascii=False, indent=1), encoding="utf-8")
    (data / "leads.json").write_text(json.dumps(LEADS, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote data/inventory.json and data/leads.json")


if __name__ == "__main__":
    main()
