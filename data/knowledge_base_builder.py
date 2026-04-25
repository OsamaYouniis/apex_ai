"""
data/knowledge_base_builder.py
================================
APEX AI — Knowledge Base Builder v2

Builds the complete knowledge base for the RAG fitness coach.

Sources:
  ✅ WHO Physical Activity Guidelines PDF (free, CC BY-NC-SA)
  ✅ NIH Office of Dietary Supplements  (public domain, US govt)
  ✅ ISSN Position Stands               (Creative Commons CC-BY)
  ✅ PubMed Central papers              (open access)
  ✅ USDA FoodData Central              (public domain, already in repo)
  ✅ wger.de Exercise Database API      (open source, no key required)
  ✅ TheMealDB Recipe API               (free tier, no key required)
  ✅ Generated Meal Plan Templates      (LLM-generated, stored as JSON)
  ✅ Curated fitness docs               (hand-authored, fills topic gaps)

Usage:
    python data/knowledge_base_builder.py              # full build
    python data/knowledge_base_builder.py --source wger
    python data/knowledge_base_builder.py --source meals
    python data/knowledge_base_builder.py --source meal_plans
    python data/knowledge_base_builder.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kb_builder")

RAW_DIR = Path("knowledge_base/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ApexAI-KB-Builder/2.0; "
        "fitness research; contact: your@email.com)"
    ),
    "Accept": "application/json, text/html",
}

REQUEST_DELAY = 1.2
TIMEOUT = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
    def handle_data(self, data):
        self._parts.append(data)
    def get_text(self):
        return " ".join(self._parts).strip()


def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_text()
    return re.sub(r"\s{2,}", " ", text).strip()


def _get(url: str, stream: bool = False, params: dict = None) -> Optional[requests.Response]:
    try:
        r = requests.get(
            url, headers=HEADERS, timeout=TIMEOUT,
            stream=stream, params=params
        )
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        log.warning(f"HTTP {e.response.status_code} — {url}")
    except requests.RequestException as e:
        log.warning(f"Request failed — {url} — {e}")
    return None


def _save(content: str, filename: str, overwrite: bool = False) -> bool:
    dest = RAW_DIR / filename
    if dest.exists() and not overwrite:
        log.info(f"  ✔  {filename} already exists")
        return True
    dest.write_text(content, encoding="utf-8")
    log.info(f"  ✅ Saved {filename} ({len(content):,} chars)")
    return True


def _save_json(data: dict | list, filename: str, overwrite: bool = False) -> bool:
    dest = RAW_DIR / filename
    if dest.exists() and not overwrite:
        log.info(f"  ✔  {filename} already exists")
        return True
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  ✅ Saved {filename}")
    return True


def _save_pdf(url: str, filename: str) -> bool:
    dest = RAW_DIR / filename
    if dest.exists():
        log.info(f"  ✔  {filename} already exists")
        return True
    log.info(f"  ⬇  Downloading {filename} …")
    r = _get(url, stream=True)
    if not r:
        return False
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"  ✅ Saved {filename} ({dest.stat().st_size // 1024} KB)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — WHO Physical Activity Guidelines
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_who_guidelines() -> int:
    log.info("📥 WHO Physical Activity Guidelines 2020")
    urls = [
        "https://apps.who.int/iris/bitstream/handle/10665/336656/9789240015128-eng.pdf",
        "https://iris.who.int/bitstream/handle/10665/336656/9789240015128-eng.pdf",
    ]
    for url in urls:
        if _save_pdf(url, "who_physical_activity_guidelines_2020.pdf"):
            return 1
        time.sleep(REQUEST_DELAY)
    log.warning("  ⚠️  Manual download: https://www.who.int/publications/i/item/9789240015128")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — NIH Office of Dietary Supplements (public domain)
# ═══════════════════════════════════════════════════════════════════════════════

NIH_FACTSHEETS = [
    ("ExerciseAndAthleticPerformance-HealthProfessional", "nih_exercise_athletic_performance.txt"),
    ("VitaminD-HealthProfessional",                       "nih_vitamin_d.txt"),
    ("Omega3FattyAcids-HealthProfessional",               "nih_omega3.txt"),
    ("Magnesium-HealthProfessional",                      "nih_magnesium.txt"),
    ("Iron-HealthProfessional",                           "nih_iron.txt"),
    ("Zinc-HealthProfessional",                           "nih_zinc.txt"),
    ("WeightLoss-HealthProfessional",                     "nih_weight_loss_supplements.txt"),
]


def fetch_nih_factsheets() -> int:
    log.info("📥 NIH Office of Dietary Supplements")
    saved = 0
    for slug, filename in NIH_FACTSHEETS:
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue
        url = f"https://ods.od.nih.gov/factsheets/{slug}/"
        r = _get(url)
        if not r:
            time.sleep(REQUEST_DELAY); continue
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.select("nav, footer, script, style, .sidebar, #breadcrumb"):
            tag.decompose()
        main = soup.find("main") or soup.body
        parts = [f"SOURCE: NIH Office of Dietary Supplements\nURL: {url}\nLICENSE: US Government public domain\n{'='*60}\n"]
        for el in main.find_all(["h1","h2","h3","h4","p","li"]):
            text = el.get_text(separator=" ", strip=True)
            if len(text) < 10: continue
            if el.name in ("h1","h2"): parts.append(f"\n## {text}\n")
            elif el.name in ("h3","h4"): parts.append(f"\n### {text}\n")
            elif el.name == "li": parts.append(f"• {text}")
            else: parts.append(text)
        _save(re.sub(r"\n{3,}", "\n\n", "\n".join(parts)), filename)
        saved += 1
        time.sleep(REQUEST_DELAY)
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — ISSN Position Stands (CC-BY)
# ═══════════════════════════════════════════════════════════════════════════════

ISSN_ARTICLES = [
    ("https://jissn.biomedcentral.com/articles/10.1186/s12970-017-0173-z", "issn_creatine.txt"),
    ("https://jissn.biomedcentral.com/articles/10.1186/s12970-017-0177-8", "issn_protein_exercise.txt"),
    ("https://jissn.biomedcentral.com/articles/10.1186/s12970-015-0090-y", "issn_beta_alanine.txt"),
    ("https://jissn.biomedcentral.com/articles/10.1186/1550-2783-7-5",     "issn_caffeine.txt"),
    ("https://jissn.biomedcentral.com/articles/10.1186/s12970-017-0174-y", "issn_calorie_macros.txt"),
    ("https://jissn.biomedcentral.com/articles/10.1186/s12970-018-0242-y", "issn_diets_body_composition.txt"),
]


def fetch_issn_position_stands() -> int:
    log.info("📥 ISSN Position Stands (BioMed Central CC-BY)")
    saved = 0
    for url, filename in ISSN_ARTICLES:
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue
        r = _get(url)
        if not r:
            time.sleep(REQUEST_DELAY); continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.find("h1", {"class": "c-article-title"}) or soup.find("h1"))
        title_text = title.get_text(strip=True) if title else filename
        body = soup.find("div", {"class": "c-article-body"}) or soup.find("main")
        parts = [f"# {title_text}\nSource: JISSN\nURL: {url}\nLicense: CC-BY 4.0\n{'='*60}\n"]
        if body:
            for el in body.find_all(["h2","h3","p","li"]):
                parent_cls = " ".join(el.parent.get("class", []))
                if "reference" in parent_cls.lower(): continue
                text = el.get_text(separator=" ", strip=True)
                if len(text) < 15: continue
                if el.name == "h2": parts.append(f"\n## {text}\n")
                elif el.name == "h3": parts.append(f"\n### {text}\n")
                elif el.name == "li": parts.append(f"• {text}")
                else: parts.append(text)
        _save(re.sub(r"\n{3,}", "\n\n", "\n".join(parts)), filename)
        saved += 1
        time.sleep(REQUEST_DELAY)
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — PubMed Central open-access papers
# ═══════════════════════════════════════════════════════════════════════════════

PMC_ARTICLES = [
    ("PMC6950543", "pmc_hypertrophy_mechanisms.txt"),
    ("PMC6279907", "pmc_resistance_training_dose_response.txt"),
    ("PMC5946208", "pmc_protein_timing.txt"),
    ("PMC6566799", "pmc_sleep_performance.txt"),
    ("PMC3943438", "pmc_hiit_vs_liss.txt"),
    ("PMC6315940", "pmc_intermittent_fasting.txt"),
]


def fetch_pmc_papers() -> int:
    log.info("📥 PubMed Central open-access papers")
    saved = 0
    for pmc_id, filename in PMC_ARTICLES:
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
        r = _get(url)
        if not r:
            time.sleep(REQUEST_DELAY); continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.find("h1", {"class": "content-title"}) or soup.find("h1")
        title_text = title.get_text(strip=True) if title else pmc_id
        body = soup.find("article") or soup.find("main")
        parts = [f"# {title_text}\nSource: PubMed Central ({pmc_id})\nURL: {url}\nLicense: Open Access\n{'='*60}\n"]
        if body:
            for el in body.find_all(["h2","h3","p","li"]):
                if el.find_parent(id=lambda x: x and "ref" in x.lower()): continue
                text = el.get_text(separator=" ", strip=True)
                if len(text) < 20: continue
                if el.name == "h2": parts.append(f"\n## {text}\n")
                elif el.name == "h3": parts.append(f"\n### {text}\n")
                elif el.name == "li": parts.append(f"• {text}")
                else: parts.append(text)
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(parts))
        if len(content) > 500:
            _save(content, filename); saved += 1
        time.sleep(REQUEST_DELAY)
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5 — USDA FoodData Central (already in repo, needs conversion)
# ═══════════════════════════════════════════════════════════════════════════════

KEY_NUTRIENTS = {
    "Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference",
    "Fiber, total dietary", "Sugars, total including NLEA",
    "Calcium, Ca", "Iron, Fe", "Magnesium, Mg", "Potassium, K",
    "Sodium, Na", "Zinc, Zn", "Vitamin C, total ascorbic acid",
    "Vitamin D (D2 + D3)", "Fatty acids, total saturated",
    "Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated",
    "Cholesterol",
}


def convert_usda_food_data(batch_size: int = 50) -> int:
    src = Path("knowledge_base/raw/FoodData_Central_foundation_food_json_2025-12-18.json")
    if not src.exists():
        log.warning("  ⚠️  USDA FoodData JSON not found — skipping"); return 0
    log.info("📥 Converting USDA FoodData Central → RAG chunks")
    with open(src) as f:
        foods = json.load(f).get("FoundationFoods", [])
    log.info(f"  Processing {len(foods)} foods …")
    saved = 0
    for i in range(0, len(foods), batch_size):
        batch = foods[i:i+batch_size]
        n = i // batch_size + 1
        filename = f"usda_foods_batch_{n:02d}.txt"
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue
        chunks = []
        for food in batch:
            name = food.get("description", "Unknown")
            cat = food.get("foodCategory", {})
            cat_name = cat.get("description", "") if isinstance(cat, dict) else str(cat)
            lines = [f"FOOD: {name}", f"Category: {cat_name}", "Nutrients per 100g:"]
            for n_item in food.get("foodNutrients", []):
                nutrient = n_item.get("nutrient", {})
                nname = nutrient.get("name", "")
                if nname not in KEY_NUTRIENTS: continue
                amount = n_item.get("amount")
                unit = nutrient.get("unitName", "")
                if amount is not None:
                    lines.append(f"  • {nname}: {amount} {unit}")
            portions = food.get("foodPortions", [])
            if portions:
                lines.append("Common portions:")
                for p in portions[:3]:
                    g = p.get("gramWeight", 0)
                    desc = p.get("portionDescription", p.get("modifier", "serving"))
                    if g: lines.append(f"  • {desc}: {g}g")
            chunks.append("\n".join(lines))
        header = f"SOURCE: USDA FoodData Central\nLICENSE: Public domain\nBatch {n} (foods {i+1}–{i+len(batch)})\n{'='*60}\n\n"
        _save(header + "\n\n---\n\n".join(chunks), filename)
        saved += 1
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 6 — wger.de Exercise Database (open source, no API key)
# License: AGPL — data is freely usable
# ═══════════════════════════════════════════════════════════════════════════════

# Sets/reps/rest prescriptions by goal — based on NSCA/ACSM guidelines
PRESCRIPTIONS = {
    "strength":     {"compound": {"sets": 4, "reps": "3–5",   "rest_sec": 180, "tempo": "2-0-X-0"},
                     "isolation": {"sets": 3, "reps": "5–8",   "rest_sec": 120, "tempo": "2-0-1-0"}},
    "hypertrophy":  {"compound": {"sets": 4, "reps": "8–12",  "rest_sec": 90,  "tempo": "3-0-1-0"},
                     "isolation": {"sets": 3, "reps": "10–15", "rest_sec": 60,  "tempo": "2-0-1-0"}},
    "fat_loss":     {"compound": {"sets": 3, "reps": "10–15", "rest_sec": 60,  "tempo": "2-0-1-0"},
                     "isolation": {"sets": 3, "reps": "12–15", "rest_sec": 45,  "tempo": "2-0-1-0"}},
    "endurance":    {"compound": {"sets": 3, "reps": "15–20", "rest_sec": 45,  "tempo": "2-0-1-0"},
                     "isolation": {"sets": 2, "reps": "15–20", "rest_sec": 30,  "tempo": "2-0-1-0"}},
    "general":      {"compound": {"sets": 3, "reps": "10–12", "rest_sec": 75,  "tempo": "2-0-1-0"},
                     "isolation": {"sets": 3, "reps": "12–15", "rest_sec": 60,  "tempo": "2-0-1-0"}},
}

# Progressive overload guidelines by fitness level
PROGRESSION = {
    "beginner":     {"lower_body_weight_increase_kg": 2.5, "upper_body_weight_increase_kg": 1.25,
                     "deload_every_weeks": 6,  "volume_progression": "add 1 set after 2 weeks at top of rep range"},
    "intermediate": {"lower_body_weight_increase_kg": 2.5, "upper_body_weight_increase_kg": 1.25,
                     "deload_every_weeks": 4,  "volume_progression": "add 1 set after 3 weeks or use DUP"},
    "advanced":     {"lower_body_weight_increase_kg": 1.25,"upper_body_weight_increase_kg": 0.5,
                     "deload_every_weeks": 4,  "volume_progression": "block periodization recommended"},
}

# Compound vs isolation classification by muscle group
COMPOUND_CATEGORIES = {8, 9, 10, 12}   # chest, back, shoulders, legs
ISOLATION_CATEGORIES = {11, 13, 14}    # arms, core, calves

# Equipment mapping (wger IDs → human names)
EQUIPMENT_MAP = {
    1: "Barbell", 2: "SZ-Bar", 3: "Dumbbell", 4: "Machine",
    5: "Cables", 6: "Kettlebell", 7: "Bands", 8: "Foam Roll",
    9: "None (bodyweight)", 10: "Pull-up Bar",
}

# Category mapping (wger IDs → human names)
CATEGORY_MAP = {
    8: "Chest", 9: "Back", 10: "Shoulders", 11: "Arms",
    12: "Legs", 13: "Core / Abs", 14: "Calves", 15: "Cardio",
}


def _fetch_all_wger_exercises() -> list[dict]:
    """Paginate through the wger exercise API and return all English exercises."""
    base_url = "https://wger.de/api/v2/exercise/"
    all_exercises = []
    url = base_url
    params = {"format": "json", "language": 2, "limit": 100, "offset": 0}

    while url:
        log.info(f"  ↳ Fetching offset={params.get('offset', '?')} …")
        r = _get(url, params=params if "offset" in params else None)
        if not r:
            break
        data = r.json()
        all_exercises.extend(data.get("results", []))
        next_url = data.get("next")
        if next_url:
            url = next_url
            params = {}   # next URL already has all params embedded
        else:
            break
        time.sleep(REQUEST_DELAY)

    return all_exercises


def _format_exercise_chunk(ex: dict) -> str:
    """Convert one wger exercise into a detailed RAG text chunk."""
    name = ex.get("name", "Unknown exercise")
    description = _strip_html(ex.get("description", ""))
    
    category = ex.get("category", {})
    cat_id = category.get("id", 0) if isinstance(category, dict) else 0
    cat_name = CATEGORY_MAP.get(cat_id, category.get("name", "General") if isinstance(category, dict) else "General")

    muscles = ex.get("muscles", [])
    muscles_secondary = ex.get("muscles_secondary", [])
    equipment_list = ex.get("equipment", [])

    primary_muscles = [m.get("name_en", m.get("name", "")) for m in muscles if isinstance(m, dict)]
    secondary_muscles = [m.get("name_en", m.get("name", "")) for m in muscles_secondary if isinstance(m, dict)]
    equipment_names = []
    for eq in equipment_list:
        if isinstance(eq, dict):
            eq_id = eq.get("id", 0)
            equipment_names.append(EQUIPMENT_MAP.get(eq_id, eq.get("name", "Unknown")))

    is_compound = cat_id in COMPOUND_CATEGORIES
    exercise_type = "compound" if is_compound else "isolation"

    lines = [
        f"EXERCISE: {name}",
        f"Muscle Group: {cat_name}",
        f"Type: {exercise_type.capitalize()}",
        f"Primary Muscles: {', '.join(primary_muscles) if primary_muscles else 'See description'}",
    ]
    if secondary_muscles:
        lines.append(f"Secondary Muscles: {', '.join(secondary_muscles)}")
    if equipment_names:
        lines.append(f"Equipment: {', '.join(equipment_names)}")
    if description:
        lines.append(f"\nExecution:\n{description}")

    lines.append("\nPrescription by goal:")
    for goal, presc in PRESCRIPTIONS.items():
        p = presc[exercise_type]
        lines.append(
            f"  • {goal.capitalize()}: {p['sets']} sets × {p['reps']} reps | "
            f"Rest: {p['rest_sec']}s | Tempo: {p['tempo']}"
        )

    lines.append("\nProgressive overload:")
    for level, prog in PROGRESSION.items():
        key = "lower_body_weight_increase_kg" if cat_id == 12 else "upper_body_weight_increase_kg"
        lines.append(
            f"  • {level.capitalize()}: +{prog[key]}kg when top of rep range hit | "
            f"Deload every {prog['deload_every_weeks']} weeks"
        )

    return "\n".join(lines)


def fetch_wger_exercises() -> int:
    """
    Fetch all exercises from wger.de, enrich with prescription data,
    and save as RAG-friendly text chunks batched by muscle group.
    """
    log.info("📥 wger.de Exercise Database (open source)")

    # Check if all batch files already exist
    muscle_groups = list(CATEGORY_MAP.values()) + ["Other"]
    existing = all((RAW_DIR / f"exercises_{g.lower().replace(' / ', '_').replace(' ', '_')}.txt").exists()
                   for g in muscle_groups)
    if existing:
        log.info("  ✔  All exercise files already exist"); return len(muscle_groups)

    exercises = _fetch_all_wger_exercises()
    if not exercises:
        log.warning("  ⚠️  Could not fetch from wger.de — generating curated exercise database instead")
        return _generate_curated_exercises()

    log.info(f"  Fetched {len(exercises)} exercises — organising by muscle group …")

    # Group by category
    groups: dict[str, list[str]] = {}
    for ex in exercises:
        cat = ex.get("category", {})
        cat_id = cat.get("id", 0) if isinstance(cat, dict) else 0
        group = CATEGORY_MAP.get(cat_id, "Other")
        chunk = _format_exercise_chunk(ex)
        if len(chunk) > 100:
            groups.setdefault(group, []).append(chunk)

    saved = 0
    for group, chunks in groups.items():
        filename = f"exercises_{group.lower().replace(' / ', '_').replace(' ', '_')}.txt"
        header = (
            f"SOURCE: wger.de Exercise Database\n"
            f"LICENSE: Open Source (AGPL)\n"
            f"Muscle Group: {group} — {len(chunks)} exercises\n"
            f"{'='*60}\n\n"
        )
        _save(header + "\n\n---\n\n".join(chunks), filename)
        saved += 1

    log.info(f"  Saved {saved} exercise files covering {sum(len(c) for c in groups.values())} exercises")
    return saved


def _generate_curated_exercises() -> int:
    """
    Fallback: generate a curated exercise database when wger.de is unreachable.
    Covers all major muscle groups with 8-12 exercises each.
    """
    log.info("  Generating curated exercise database as fallback …")

    exercises_db = {
        "chest": [
            ("Barbell Bench Press", "compound", "Barbell",
             "Lie flat on bench. Grip bar slightly wider than shoulder-width. "
             "Lower bar to mid-chest with elbows at ~45°. Press back up to full extension. "
             "Keep feet flat, slight arch in lower back, shoulder blades retracted.",
             ["Pectoralis major"], ["Anterior deltoid", "Triceps brachii"]),
            ("Incline Dumbbell Press", "compound", "Dumbbell",
             "Set bench to 30-45°. Press dumbbells from shoulder height to lockout. "
             "Targets upper chest. Control the descent.",
             ["Pectoralis major (upper)", "Anterior deltoid"], ["Triceps"]),
            ("Cable Fly", "isolation", "Cables",
             "Stand between cable stations. Pull handles together in a wide arc, "
             "meeting at chest height. Maintain slight elbow bend throughout.",
             ["Pectoralis major"], ["Anterior deltoid"]),
            ("Push-up", "compound", "Bodyweight",
             "Hands slightly wider than shoulder-width. Body forms straight line. "
             "Lower chest to within 2cm of floor. Push back up. "
             "Regress: knees down. Progress: feet elevated, archer push-up.",
             ["Pectoralis major"], ["Triceps", "Anterior deltoid", "Core"]),
            ("Dumbbell Pullover", "isolation", "Dumbbell",
             "Lie across bench. Hold dumbbell overhead with both hands. "
             "Lower behind head in arc, return to start. Stretches chest and lats.",
             ["Pectoralis major", "Serratus anterior"], ["Latissimus dorsi"]),
        ],
        "back": [
            ("Deadlift", "compound", "Barbell",
             "Bar over mid-foot. Hinge at hips, grip just outside legs. "
             "Brace core, pull bar up by driving floor away. Lock hips and knees together at top. "
             "Keep bar close to body throughout. Neutral spine mandatory.",
             ["Erector spinae", "Gluteus maximus", "Hamstrings"],
             ["Latissimus dorsi", "Trapezius", "Forearms"]),
            ("Barbell Row", "compound", "Barbell",
             "Hinge to ~45°. Pull bar to lower ribcage, leading with elbows. "
             "Squeeze shoulder blades at top. Control descent.",
             ["Latissimus dorsi", "Rhomboids", "Middle trapezius"],
             ["Biceps brachii", "Rear deltoid"]),
            ("Pull-up / Chin-up", "compound", "Pull-up Bar",
             "Dead hang from bar. Pull until chin clears bar. "
             "Pull-up = overhand (more lats). Chin-up = underhand (more biceps). "
             "Progress: weighted belt. Regress: band-assisted or lat pulldown.",
             ["Latissimus dorsi", "Biceps brachii"],
             ["Rear deltoid", "Rhomboids", "Core"]),
            ("Seated Cable Row", "compound", "Cables",
             "Sit upright at cable station. Pull handle to lower ribcage. "
             "Elbows stay close to body. Avoid rounding forward.",
             ["Latissimus dorsi", "Rhomboids"], ["Biceps", "Erector spinae"]),
            ("Face Pull", "isolation", "Cables",
             "Set cable at head height. Pull rope to face, separating hands at end. "
             "External rotation at end range. Essential for shoulder health.",
             ["Rear deltoid", "Rotator cuff", "Middle trapezius"], ["Rhomboids"]),
        ],
        "shoulders": [
            ("Overhead Press", "compound", "Barbell",
             "Bar at shoulder height in rack. Press overhead to full lockout. "
             "Bar travels in slight arc around face. Brace core throughout. "
             "Dumbbell variation: more range of motion, greater stability demand.",
             ["Anterior deltoid", "Lateral deltoid"],
             ["Triceps", "Upper trapezius", "Core"]),
            ("Lateral Raise", "isolation", "Dumbbell",
             "Stand with dumbbells at sides. Raise arms to shoulder height in arc. "
             "Slight forward lean and thumb-down position to bias lateral head. "
             "Control descent — 3 seconds down.",
             ["Lateral deltoid"], ["Supraspinatus"]),
            ("Arnold Press", "compound", "Dumbbell",
             "Start with palms facing you at shoulder height. "
             "Rotate palms outward as you press overhead. Reverse on descent. "
             "Full range of motion hits all three deltoid heads.",
             ["Anterior deltoid", "Lateral deltoid", "Posterior deltoid"],
             ["Triceps", "Upper trapezius"]),
            ("Rear Delt Fly", "isolation", "Dumbbell",
             "Hinge forward 45-90°. Raise dumbbells in wide arc to shoulder height. "
             "Pinch shoulder blades. Critical for posture and shoulder balance.",
             ["Posterior deltoid", "Rhomboids"], ["Middle trapezius"]),
        ],
        "arms": [
            ("Barbell Curl", "isolation", "Barbell",
             "Stand with barbell, shoulder-width underhand grip. "
             "Curl to shoulder height, squeeze bicep. Lower in 3 seconds. "
             "Keep elbows fixed at sides — no swinging.",
             ["Biceps brachii", "Brachialis"], ["Forearm flexors"]),
            ("Tricep Pushdown", "isolation", "Cables",
             "Set cable high. Push bar/rope down to full extension. "
             "Elbows stay at sides. Squeeze triceps at lockout.",
             ["Triceps brachii (lateral + medial head)"], ["Anconeus"]),
            ("Hammer Curl", "isolation", "Dumbbell",
             "Neutral grip (thumbs up). Curl dumbbells to shoulder. "
             "Targets brachialis — adds arm thickness.",
             ["Brachialis", "Brachioradialis"], ["Biceps brachii"]),
            ("Skull Crusher", "isolation", "Barbell",
             "Lie on bench. Lower bar toward forehead by hinging elbows. "
             "Press back to lockout. Also called lying tricep extension.",
             ["Triceps brachii (long head)"], ["Anconeus"]),
            ("Preacher Curl", "isolation", "Barbell",
             "Use preacher bench. Curl from full extension to peak contraction. "
             "Eliminates momentum. Excellent for peak bicep development.",
             ["Biceps brachii (short head)"], ["Brachialis"]),
        ],
        "legs": [
            ("Barbell Squat", "compound", "Barbell",
             "Bar on upper traps (high bar) or rear delts (low bar). "
             "Feet shoulder-width, toes slightly out. Break at hips and knees simultaneously. "
             "Descend until hips below knees. Drive through full foot on ascent. "
             "Keep chest up, knees tracking toes.",
             ["Quadriceps", "Gluteus maximus", "Hamstrings"],
             ["Adductors", "Core", "Erector spinae"]),
            ("Romanian Deadlift (RDL)", "compound", "Barbell",
             "Start standing with barbell. Hinge at hips, pushing them back. "
             "Lower bar along legs until strong stretch in hamstrings (~shin level). "
             "Drive hips forward to return. Knees slightly bent throughout.",
             ["Hamstrings", "Gluteus maximus"], ["Erector spinae", "Adductors"]),
            ("Leg Press", "compound", "Machine",
             "Feet shoulder-width on platform. Lower until 90° knee angle. "
             "Press to near-lockout (don't lock knees). "
             "High foot placement = more glutes/hamstrings. "
             "Low foot placement = more quads.",
             ["Quadriceps", "Gluteus maximus"], ["Hamstrings", "Adductors"]),
            ("Bulgarian Split Squat", "compound", "Dumbbell",
             "Rear foot elevated on bench. Front foot forward enough to maintain "
             "vertical shin. Lower rear knee toward floor. Loaded with dumbbells or barbell. "
             "Excellent unilateral leg developer.",
             ["Quadriceps", "Gluteus maximus"], ["Hamstrings", "Adductors", "Core"]),
            ("Glute Bridge / Hip Thrust", "isolation", "Barbell",
             "Back on bench, barbell across hips. Drive hips up to full extension. "
             "Squeeze glutes at top. Hold 1 second. Best exercise for glute isolation.",
             ["Gluteus maximus", "Gluteus medius"], ["Hamstrings", "Core"]),
            ("Leg Curl", "isolation", "Machine",
             "Lie prone on machine. Curl heels toward glutes. "
             "Slow eccentric (3 seconds) for maximum hamstring development.",
             ["Hamstrings (biceps femoris)"], ["Gastrocnemius"]),
            ("Calf Raise", "isolation", "Machine",
             "Full range of motion — deep stretch at bottom, full plantar flexion at top. "
             "3-second hold at top. Seat calf raises target soleus; "
             "standing targets gastrocnemius.",
             ["Gastrocnemius", "Soleus"], []),
        ],
        "core": [
            ("Plank", "isolation", "Bodyweight",
             "Forearms on floor. Body straight from head to heels. "
             "Brace like expecting a punch. Breathe normally. "
             "Do NOT hold breath. Progress: weighted vest, feet elevated, single-arm.",
             ["Transverse abdominis", "Rectus abdominis"], ["Glutes", "Shoulder stabilizers"]),
            ("Dead Bug", "isolation", "Bodyweight",
             "Lie on back. Arms straight up, hips/knees at 90°. "
             "Lower opposite arm and leg simultaneously while pressing lower back into floor. "
             "Return. Alternate sides. Best anti-extension core exercise.",
             ["Transverse abdominis", "Rectus abdominis"], ["Hip flexors", "Shoulder stabilizers"]),
            ("Cable Crunch", "isolation", "Cables",
             "Kneel facing cable machine. Rope attachment at top. "
             "Crunch elbows to knees, rounding spine. "
             "Weight-loadable — allows progressive overload for abs.",
             ["Rectus abdominis"], ["Obliques"]),
            ("Ab Wheel Rollout", "isolation", "Bodyweight",
             "Kneeling on floor. Roll wheel forward until body nearly parallel to ground. "
             "Pull back using abs. Most challenging anti-extension ab exercise.",
             ["Rectus abdominis", "Transverse abdominis"], ["Latissimus dorsi", "Shoulders"]),
            ("McGill Curl-up", "isolation", "Bodyweight",
             "Lie flat. One knee bent. Place hands under lower back. "
             "Lift only head and shoulders, maintaining neutral spine. "
             "NOT a full crunch — preserves lumbar spine.",
             ["Rectus abdominis"], ["Neck flexors"]),
        ],
    }

    saved = 0
    for muscle_group, exercises in exercises_db.items():
        filename = f"exercises_{muscle_group}.txt"
        chunks = []
        for (name, ex_type, equipment, execution, primary, secondary) in exercises:
            cat_id = {"chest": 8, "back": 9, "shoulders": 10,
                      "arms": 11, "legs": 12, "core": 13}.get(muscle_group, 0)
            lines = [
                f"EXERCISE: {name}",
                f"Muscle Group: {muscle_group.capitalize()}",
                f"Type: {ex_type.capitalize()}",
                f"Primary Muscles: {', '.join(primary)}",
            ]
            if secondary:
                lines.append(f"Secondary Muscles: {', '.join(secondary)}")
            lines.append(f"Equipment: {equipment}")
            lines.append(f"\nExecution:\n{execution}")
            lines.append("\nPrescription by goal:")
            presc = PRESCRIPTIONS.get("hypertrophy", {})
            for goal, presc_data in PRESCRIPTIONS.items():
                p = presc_data[ex_type]
                lines.append(
                    f"  • {goal.capitalize()}: {p['sets']} sets × {p['reps']} reps | "
                    f"Rest: {p['rest_sec']}s | Tempo: {p['tempo']}"
                )
            lines.append("\nProgressive overload:")
            for level, prog in PROGRESSION.items():
                key = "lower_body_weight_increase_kg" if muscle_group == "legs" else "upper_body_weight_increase_kg"
                lines.append(
                    f"  • {level.capitalize()}: +{prog[key]}kg when top of rep range hit | "
                    f"Deload every {prog['deload_every_weeks']} weeks"
                )
            chunks.append("\n".join(lines))

        header = (
            f"SOURCE: APEX AI Curated Exercise Database\n"
            f"BASED ON: NSCA Essentials of Strength & Conditioning\n"
            f"Muscle Group: {muscle_group.capitalize()} — {len(chunks)} exercises\n"
            f"{'='*60}\n\n"
        )
        _save(header + "\n\n---\n\n".join(chunks), filename)
        saved += 1

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 7 — TheMealDB Recipe API (free tier, no key)
# ═══════════════════════════════════════════════════════════════════════════════

# Fitness-relevant meal categories to fetch
FITNESS_CATEGORIES = [
    "Chicken", "Beef", "Seafood", "Vegetarian", "Vegan",
    "Pasta", "Side", "Breakfast",
]

# Macro estimation per 100g (approximate, by food type)
MACRO_ESTIMATES = {
    "chicken": {"protein": 31, "carbs": 0,  "fat": 3.6, "kcal": 165},
    "beef":    {"protein": 26, "carbs": 0,  "fat": 15,  "kcal": 250},
    "salmon":  {"protein": 25, "carbs": 0,  "fat": 13,  "kcal": 208},
    "rice":    {"protein": 2.7,"carbs": 28, "fat": 0.3, "kcal": 130},
    "egg":     {"protein": 13, "carbs": 1,  "fat": 11,  "kcal": 155},
    "oats":    {"protein": 17, "carbs": 66, "fat": 7,   "kcal": 389},
    "broccoli":{"protein": 2.8,"carbs": 7,  "fat": 0.4, "kcal": 34},
    "potato":  {"protein": 2,  "carbs": 17, "fat": 0.1, "kcal": 77},
}


def _format_meal_chunk(meal: dict) -> str:
    """Convert one TheMealDB meal into a RAG-friendly text chunk."""
    name = meal.get("strMeal", "Unknown meal")
    category = meal.get("strCategory", "")
    area = meal.get("strArea", "")
    instructions = meal.get("strInstructions", "")
    if len(instructions) > 800:
        instructions = instructions[:800] + "…"

    # Extract ingredients
    ingredients = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}")or "").strip()
        meas = (meal.get(f"strMeasure{i}")or "").strip()
        if ing:
            ingredients.append(f"{meas} {ing}".strip() if meas else ing)

    # Estimate dominant macro profile
    ing_text = " ".join(ingredients).lower()
    is_high_protein = any(w in ing_text for w in ["chicken", "beef", "salmon", "tuna", "egg", "turkey", "tofu"])
    is_high_carb = any(w in ing_text for w in ["rice", "pasta", "bread", "oat", "potato", "noodle"])
    is_high_fat = any(w in ing_text for w in ["oil", "butter", "avocado", "cheese", "nuts", "cream"])

    profile_parts = []
    if is_high_protein: profile_parts.append("high-protein")
    if is_high_carb:    profile_parts.append("high-carb")
    if is_high_fat:     profile_parts.append("high-fat")
    macro_profile = ", ".join(profile_parts) if profile_parts else "mixed macros"

    tags = meal.get("strTags", "") or ""
    dietary_tags = []
    if "vegetarian" in tags.lower() or category.lower() == "vegetarian":
        dietary_tags.append("vegetarian")
    if "vegan" in tags.lower() or category.lower() == "vegan":
        dietary_tags.append("vegan")

    lines = [
        f"RECIPE: {name}",
        f"Category: {category}" + (f" | Cuisine: {area}" if area else ""),
        f"Macro profile: {macro_profile}",
        f"Dietary: {', '.join(dietary_tags) if dietary_tags else 'omnivore'}",
        f"\nIngredients ({len(ingredients)} items):",
    ]
    for ing in ingredients:
        lines.append(f"  • {ing}")
    if instructions:
        lines.append(f"\nInstructions:\n{instructions}")

    lines.append(f"\nFitness context:")
    if is_high_protein:
        lines.append("  • Good post-workout meal — high protein supports muscle protein synthesis")
    if is_high_carb:
        lines.append("  • Good pre-workout meal — carbohydrates fuel training performance")
    if "breakfast" in category.lower():
        lines.append("  • Suitable as breakfast — fuel for morning training")

    return "\n".join(lines)


def fetch_themealdb_recipes() -> int:
    """
    Fetch fitness-relevant recipes from TheMealDB free API.
    No API key required. Saves by category.
    """
    log.info("📥 TheMealDB Recipe Database (free API)")
    saved = 0
    base_url = "https://www.themealdb.com/api/json/v1/1"

    for category in FITNESS_CATEGORIES:
        filename = f"recipes_{category.lower()}.txt"
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue

        log.info(f"  ↳ Fetching {category} recipes …")
        r = _get(f"{base_url}/filter.php", params={"c": category})
        if not r:
            time.sleep(REQUEST_DELAY); continue

        meal_list = r.json().get("meals", []) or []
        chunks = []

        for i, meal_stub in enumerate(meal_list[:20]):  # max 20 per category
            meal_id = meal_stub.get("idMeal")
            if not meal_id:
                continue
            time.sleep(0.5)
            detail_r = _get(f"{base_url}/lookup.php", params={"i": meal_id})
            if not detail_r:
                continue
            meals = detail_r.json().get("meals", [])
            if meals:
                chunk = _format_meal_chunk(meals[0])
                if len(chunk) > 100:
                    chunks.append(chunk)

        if chunks:
            header = (
                f"SOURCE: TheMealDB\nURL: https://www.themealdb.com\n"
                f"LICENSE: Free to use\n"
                f"Category: {category} — {len(chunks)} recipes\n"
                f"{'='*60}\n\n"
            )
            _save(header + "\n\n---\n\n".join(chunks), filename)
            saved += 1
        time.sleep(REQUEST_DELAY)

    return saved


def _generate_curated_recipes() -> int:
    """Fallback: generate structured meal recipes when TheMealDB is unreachable."""
    log.info("  Generating curated meal database as fallback …")

    MEALS = {
        "recipes_high_protein.txt": [
            {
                "name": "Classic Chicken and Rice Bowl",
                "macros_per_serving": {"protein_g": 52, "carbs_g": 45, "fat_g": 8, "kcal": 464},
                "ingredients": [
                    "200g chicken breast (skinless)", "150g cooked white rice",
                    "100g broccoli florets", "1 tbsp olive oil",
                    "Garlic powder, paprika, salt, pepper to taste"
                ],
                "instructions": (
                    "1. Season chicken with garlic powder, paprika, salt, pepper.\n"
                    "2. Cook rice according to packet (brings to 150g cooked from ~60g dry).\n"
                    "3. Heat olive oil in pan, cook chicken 6-7 min per side until 74°C internal.\n"
                    "4. Steam or microwave broccoli 3 min.\n"
                    "5. Rest chicken 5 min, slice, serve over rice with broccoli.\n"
                    "Meal prep: scales to 4 servings easily. Store 4 days in fridge."
                ),
                "fitness_context": "Ideal post-workout meal. High protein for muscle repair. "
                                   "Moderate carbs to replenish glycogen. Low fat for fast digestion.",
                "dietary": "omnivore, gluten-free, dairy-free",
            },
            {
                "name": "Greek Yogurt Protein Bowl",
                "macros_per_serving": {"protein_g": 35, "carbs_g": 40, "fat_g": 5, "kcal": 345},
                "ingredients": [
                    "300g Greek yogurt (0% fat)", "1 scoop whey protein (optional, +25g protein)",
                    "80g oats (dry weight)", "1 banana (sliced)",
                    "100g mixed berries", "1 tbsp honey", "30g granola"
                ],
                "instructions": (
                    "1. Stir protein powder into yogurt if using.\n"
                    "2. Cook oats with water/milk per packet.\n"
                    "3. Layer: oats, yogurt mix, berries, banana, granola, drizzle honey.\n"
                    "Prep time: 5 minutes."
                ),
                "fitness_context": "Excellent breakfast or post-workout meal. "
                                   "Slow + fast protein (casein + whey) for sustained MPS. "
                                   "Berries provide antioxidants for recovery.",
                "dietary": "vegetarian, gluten-free option (use GF oats)",
            },
            {
                "name": "Tuna and Avocado Rice Cakes",
                "macros_per_serving": {"protein_g": 30, "carbs_g": 20, "fat_g": 12, "kcal": 308},
                "ingredients": [
                    "1 can (160g) tuna in water (drained)", "1/2 avocado (mashed)",
                    "4 rice cakes", "1/4 red onion (diced)", "Lemon juice, salt, pepper"
                ],
                "instructions": (
                    "1. Drain tuna thoroughly.\n"
                    "2. Mash avocado with lemon juice, salt, pepper.\n"
                    "3. Mix tuna with avocado and diced onion.\n"
                    "4. Spread on rice cakes. Serve immediately.\n"
                    "Prep time: 5 minutes. No cooking required."
                ),
                "fitness_context": "Quick high-protein snack or light meal. "
                                   "Omega-3 from tuna supports recovery and inflammation reduction. "
                                   "Good pre-workout snack 1-2 hours before training.",
                "dietary": "pescatarian, gluten-free, dairy-free",
            },
        ],
        "recipes_vegetarian.txt": [
            {
                "name": "Lentil and Vegetable Curry",
                "macros_per_serving": {"protein_g": 22, "carbs_g": 58, "fat_g": 6, "kcal": 378},
                "ingredients": [
                    "200g red lentils (dry)", "400ml coconut milk (light)",
                    "400g tinned chopped tomatoes", "200g spinach",
                    "1 onion (diced)", "3 garlic cloves", "1 tsp turmeric",
                    "1 tsp cumin", "1 tsp garam masala", "300g cooked brown rice"
                ],
                "instructions": (
                    "1. Sauté onion and garlic in oil 5 min.\n"
                    "2. Add spices, cook 1 min.\n"
                    "3. Add lentils, tomatoes, coconut milk. Simmer 20 min.\n"
                    "4. Stir in spinach until wilted.\n"
                    "5. Serve over brown rice.\n"
                    "Meal prep: Makes 4 servings. Keeps 5 days refrigerated."
                ),
                "fitness_context": "Complete plant-based protein (lentils + rice = all essential amino acids). "
                                   "High fiber supports gut health. Iron-rich for endurance athletes.",
                "dietary": "vegan, gluten-free, dairy-free",
            },
            {
                "name": "Egg and Vegetable Omelette",
                "macros_per_serving": {"protein_g": 28, "carbs_g": 8, "fat_g": 18, "kcal": 310},
                "ingredients": [
                    "4 whole eggs", "100g spinach", "1/2 bell pepper (diced)",
                    "50g feta cheese (crumbled)", "1/4 onion (diced)",
                    "1 tsp olive oil", "Salt, pepper, herbs"
                ],
                "instructions": (
                    "1. Sauté pepper and onion in olive oil 3 min.\n"
                    "2. Add spinach, cook until wilted.\n"
                    "3. Beat eggs, season, pour over vegetables.\n"
                    "4. Cook on medium-low until set (~3 min).\n"
                    "5. Add feta, fold omelette, serve.\n"
                    "Prep + cook time: 10 minutes."
                ),
                "fitness_context": "High-quality complete protein. Excellent breakfast before morning training. "
                                   "Eggs provide leucine — key amino acid for muscle protein synthesis.",
                "dietary": "vegetarian, gluten-free",
            },
        ],
        "recipes_quick_prep.txt": [
            {
                "name": "Overnight Oats (Meal Prep)",
                "macros_per_serving": {"protein_g": 18, "carbs_g": 55, "fat_g": 8, "kcal": 364},
                "ingredients": [
                    "80g rolled oats", "250ml milk (or plant milk)",
                    "150g Greek yogurt", "1 tbsp chia seeds",
                    "1 tbsp almond butter", "100g berries (frozen or fresh)"
                ],
                "instructions": (
                    "1. Mix all ingredients in jar or container.\n"
                    "2. Refrigerate overnight (minimum 4 hours).\n"
                    "3. Top with fresh berries in the morning. Eat cold.\n"
                    "Prep time: 5 minutes. No cooking required.\n"
                    "Meal prep: Make 5 jars on Sunday for the week."
                ),
                "fitness_context": "Ideal pre-training breakfast. Slow-release carbs provide "
                                   "sustained energy. Protein from yogurt supports muscle maintenance.",
                "dietary": "vegetarian, can be vegan (use plant milk + vegan protein)",
            },
        ],
    }

    saved = 0
    for filename, meals in MEALS.items():
        chunks = []
        for meal in meals:
            m = meal["macros_per_serving"]
            lines = [
                f"RECIPE: {meal['name']}",
                f"Macros per serving: {m['protein_g']}g protein | "
                f"{m['carbs_g']}g carbs | {m['fat_g']}g fat | {m['kcal']} kcal",
                f"Dietary: {meal['dietary']}",
                "\nIngredients:",
            ]
            for ing in meal["ingredients"]:
                lines.append(f"  • {ing}")
            lines.append(f"\nInstructions:\n{meal['instructions']}")
            lines.append(f"\nFitness context:\n{meal['fitness_context']}")
            chunks.append("\n".join(lines))

        header = (
            f"SOURCE: APEX AI Curated Meal Database\n"
            f"BASED ON: Sports nutrition guidelines (ISSN, ACSM)\n"
            f"{len(chunks)} recipes\n"
            f"{'='*60}\n\n"
        )
        _save(header + "\n\n---\n\n".join(chunks), filename)
        saved += 1

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — Generated Meal Plan Templates
# Pre-built weekly meal schedules saved as structured JSON
# Covers: cut/bulk/maintain × calorie level × diet type
# ═══════════════════════════════════════════════════════════════════════════════

def generate_meal_plan_templates() -> int:
    """
    Generate structured weekly meal plan templates.
    Stored as JSON for programmatic use AND as text for RAG retrieval.
    """
    log.info("📥 Generating meal plan templates")

    filename_json = "meal_plan_templates.json"
    filename_txt  = "meal_plan_templates.txt"

    if (RAW_DIR / filename_json).exists() and (RAW_DIR / filename_txt).exists():
        log.info("  ✔  Meal plan templates already exist"); return 2

    # Template structure: goal → calorie_level → diet_type → 7 days
    templates = []

    goals_config = {
        "fat_loss":    [1400, 1600, 1800, 2000],
        "maintenance": [2000, 2200, 2400],
        "muscle_gain": [2500, 2800, 3200],
    }

    diet_types = {
        "standard": {
            "protein_sources": ["chicken breast", "beef mince (lean)", "salmon", "eggs", "Greek yogurt", "cottage cheese"],
            "carb_sources":    ["white rice", "brown rice", "oats", "sweet potato", "pasta", "bread (whole grain)"],
            "fat_sources":     ["olive oil", "avocado", "almonds", "peanut butter"],
            "veggies":         ["broccoli", "spinach", "mixed salad", "bell peppers", "zucchini"],
        },
        "vegetarian": {
            "protein_sources": ["eggs", "Greek yogurt", "cottage cheese", "lentils", "chickpeas", "tofu", "tempeh"],
            "carb_sources":    ["brown rice", "oats", "quinoa", "sweet potato", "whole grain bread"],
            "fat_sources":     ["olive oil", "avocado", "mixed nuts", "peanut butter", "seeds"],
            "veggies":         ["broccoli", "spinach", "lentils", "chickpeas", "mixed salad"],
        },
        "vegan": {
            "protein_sources": ["tofu", "tempeh", "lentils", "chickpeas", "black beans", "edamame", "vegan protein powder"],
            "carb_sources":    ["brown rice", "oats", "quinoa", "sweet potato", "buckwheat"],
            "fat_sources":     ["olive oil", "avocado", "mixed nuts", "tahini", "hemp seeds"],
            "veggies":         ["broccoli", "spinach", "kale", "bell peppers", "mushrooms"],
        },
    }

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_slots = ["Breakfast", "Lunch", "Dinner", "Snack"]

    for goal, calorie_levels in goals_config.items():
        for calories in calorie_levels:
            for diet_type, foods in diet_types.items():
                # Calculate macro targets
                if goal == "fat_loss":
                    protein_pct, carb_pct, fat_pct = 0.35, 0.40, 0.25
                elif goal == "muscle_gain":
                    protein_pct, carb_pct, fat_pct = 0.30, 0.45, 0.25
                else:
                    protein_pct, carb_pct, fat_pct = 0.30, 0.45, 0.25

                protein_g = round((calories * protein_pct) / 4)
                carbs_g   = round((calories * carb_pct) / 4)
                fat_g     = round((calories * fat_pct) / 9)

                per_meal_protein = round(protein_g / 4)
                per_meal_carbs   = round(carbs_g / 4)
                per_meal_kcal    = round(calories / 4)

                weekly_plan = {
                    "goal": goal,
                    "calories_target": calories,
                    "diet_type": diet_type,
                    "daily_macros": {
                        "protein_g": protein_g,
                        "carbs_g": carbs_g,
                        "fat_g": fat_g,
                    },
                    "days": {}
                }

                for day in days:
                    day_meals = {}
                    ps = foods["protein_sources"]
                    cs = foods["carb_sources"]
                    vs = foods["veggies"]

                    # Rotate through sources across days
                    day_idx = days.index(day)
                    p1 = ps[day_idx % len(ps)]
                    p2 = ps[(day_idx + 2) % len(ps)]
                    c1 = cs[day_idx % len(cs)]
                    c2 = cs[(day_idx + 1) % len(cs)]
                    v1 = vs[day_idx % len(vs)]
                    v2 = vs[(day_idx + 1) % len(vs)]

                    day_meals["Breakfast"] = {
                        "description": f"Oats with {p2} and fruit",
                        "approximate_macros": f"~{per_meal_protein}g protein, ~{per_meal_carbs}g carbs, ~{per_meal_kcal} kcal",
                        "items": [f"80g {c2} (dry)", f"200g {p2}", "1 banana or 100g berries", "250ml milk or plant milk"],
                    }
                    day_meals["Lunch"] = {
                        "description": f"{p1.title()} with {c1} and {v1}",
                        "approximate_macros": f"~{per_meal_protein}g protein, ~{per_meal_carbs}g carbs, ~{per_meal_kcal} kcal",
                        "items": [f"150-200g {p1}", f"150g cooked {c1}", f"150g {v1}", "1 tsp olive oil"],
                    }
                    day_meals["Dinner"] = {
                        "description": f"{p2.title()} stir-fry with {v2} and {c2}",
                        "approximate_macros": f"~{per_meal_protein}g protein, ~{per_meal_carbs}g carbs, ~{per_meal_kcal} kcal",
                        "items": [f"150-200g {p2}", f"150g cooked {c2}", f"200g {v2}", "herbs and spices"],
                    }
                    day_meals["Snack"] = {
                        "description": "High-protein snack",
                        "approximate_macros": f"~{round(per_meal_protein * 0.7)}g protein, ~{round(per_meal_kcal * 0.5)} kcal",
                        "items": ["150g Greek yogurt OR 30g nuts OR protein shake with milk"],
                    }
                    weekly_plan["days"][day] = day_meals

                templates.append(weekly_plan)

    # Save as JSON (for programmatic use)
    _save_json(templates, filename_json)

    # Also save as human-readable text (for RAG retrieval)
    text_parts = [
        "SOURCE: APEX AI Generated Meal Plan Templates",
        "Based on: ISSN sports nutrition guidelines",
        "Coverage: fat_loss / maintenance / muscle_gain × 3 calorie levels × 3 diet types",
        f"Total templates: {len(templates)}",
        "=" * 60, "",
    ]

    for t in templates:
        text_parts.append(
            f"MEAL PLAN: {t['goal'].replace('_',' ').title()} | "
            f"{t['calories_target']} kcal | {t['diet_type'].title()} diet"
        )
        dm = t["daily_macros"]
        text_parts.append(
            f"Daily macros: {dm['protein_g']}g protein | "
            f"{dm['carbs_g']}g carbs | {dm['fat_g']}g fat"
        )
        monday = t["days"].get("Monday", {})
        for meal_name, meal_data in monday.items():
            text_parts.append(f"  {meal_name}: {meal_data['description']}")
            text_parts.append(f"    → {meal_data['approximate_macros']}")
        text_parts.append("")

    _save("\n".join(text_parts), filename_txt)
    return 2


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — Curated fitness documents (hand-authored)
# ═══════════════════════════════════════════════════════════════════════════════

CURATED_DOCS = {
    "periodization_guide.txt": """SOURCE: APEX AI Curated Knowledge Base
TOPIC: Training Periodization
============================================================

# Periodization in Strength and Conditioning

Periodization is the systematic planning of training to reach peak performance
and prevent overtraining.

## Linear Periodization
Progressive increase in intensity, decrease in volume over time.
Weeks 1-4: hypertrophy (high volume, 8-12 reps), Weeks 5-8: strength (4-6 reps),
Weeks 9-12: power (2-4 reps). Best for beginners.

## Daily Undulating Periodization (DUP)
Variation in rep ranges within the week:
Monday 4x12 (hypertrophy), Wednesday 4x6 (strength), Friday 4x20 (endurance).
Superior hypertrophy for intermediate/advanced athletes.

## Deload Weeks
Every 4-6 weeks: reduce volume by 40-50% at same intensity.
Signs you need a deload: persistent fatigue, declining performance,
joint pain, poor sleep. Duration: 1 week.

## Volume Landmarks (Schoenfeld 2016)
Minimum effective volume: 10 sets per muscle group per week.
Maximum adaptive volume: 20 sets per muscle group per week.
Maintenance volume: 6-8 sets per muscle group per week.
""",
    "womens_training_guide.txt": """SOURCE: APEX AI Curated Knowledge Base
TOPIC: Women-Specific Training and Nutrition
============================================================

# Women-Specific Fitness

## Hormonal Considerations
Follicular phase (days 1-14): Peak strength, best for heavy training and PRs.
Luteal phase (days 15-28): Higher RPE, better endurance, increase carbs.

## Strength Training
Women benefit from higher rep ranges (10-15) for lower body.
Equal strength gains to men relative to muscle size.
Full body 3x/week highly effective.

## Key Nutrients
Iron: 18mg/day (premenopausal) — red meat, lentils + vitamin C.
Calcium: 1000-1200mg/day for bone density.
Vitamin D: 1500-2000 IU/day.
Protein: same as men — 1.6-2.2g per kg bodyweight.

## Avoid Female Athlete Triad
Low energy availability + menstrual dysfunction + low bone density.
Never restrict calories below BMR. Minimum 1600 kcal for active women.
""",
    "home_workout_guide.txt": """SOURCE: APEX AI Curated Knowledge Base
TOPIC: Home and Bodyweight Training
============================================================

# Effective Home and Bodyweight Training

## Progressive Overload Without Weights
1. Add reps (push-ups: 3x10 → 3x15 → 3x20)
2. Decrease rest time
3. Harder variations (push-up → archer → one-arm)
4. Slow tempo (3-second eccentric)
5. Single limb (squat → pistol squat)

## 3-Day Full Body Program (No Equipment)
Day A: 4x max push-ups, 4x15 glute bridges, 3x10 reverse lunges, 3x30s plank
Day B: Active recovery (30 min walk)
Day C: Same as Day A with 1 extra set or harder variation

## Cardio at Home
HIIT: 20s work / 10s rest × 8 rounds (burpees, jump squats, mountain climbers)
LISS: 30-45 min brisk walk, cycling, stair climbing
Jump rope: highest calorie burn per minute of home cardio
""",
    "injury_prevention.txt": """SOURCE: APEX AI Curated Knowledge Base
TOPIC: Common Gym Injuries — Prevention and Management
============================================================

# Injury Prevention and Management

## Shoulder Injuries
Cause: Overuse, poor form, muscle imbalance (push:pull > 1:1).
Prevention: Face pulls 3x/week, external rotation work, 1:1 push:pull ratio.
Management: Reduce overhead pressing, add band pull-aparts daily.

## Hip Flexor Strain
Cause: Tight hip flexors from sitting + explosive movements.
Prevention: Hip flexor stretching post-workout, glute activation warm-up.
Management: Avoid deep squats, add couch stretch and pigeon pose.

## Wrist Pain (Bench/Overhead Press)
Cause: Wrists extended under load.
Prevention: Keep wrists neutral (straight), use wrist wraps if needed.
Management: Reduce load, try neutral grip alternatives.

## General Injury Prevention Principles
1. Warm-up: 5 min cardio + dynamic stretching before every session
2. Progressive overload: max 5-10% volume increase per week
3. Sleep 7-9 hours — connective tissue repairs during sleep
4. Deload every 4-6 weeks
5. When in doubt, reduce load not volume
6. Sharp pain = stop immediately. Consult physiotherapist.

## RICE Protocol for Acute Injuries
Rest, Ice (15-20 min), Compression, Elevation — first 48 hours.
After 48 hours: gentle movement, heat therapy for blood flow.
""",
}


def generate_curated_docs() -> int:
    log.info("📥 Generating curated knowledge documents")
    saved = 0
    for filename, content in CURATED_DOCS.items():
        if (RAW_DIR / filename).exists():
            log.info(f"  ✔  {filename} already exists"); saved += 1; continue
        (RAW_DIR / filename).write_text(content.strip(), encoding="utf-8")
        log.info(f"  ✅ Generated {filename}")
        saved += 1
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

SOURCES = {
    "who":        ("WHO Physical Activity Guidelines",      fetch_who_guidelines),
    "nih":        ("NIH Dietary Supplements Factsheets",    fetch_nih_factsheets),
    "issn":       ("ISSN Position Stands (CC-BY)",          fetch_issn_position_stands),
    "pmc":        ("PubMed Central Open-Access Papers",     fetch_pmc_papers),
    "usda":       ("USDA FoodData Central",                 convert_usda_food_data),
    "wger":       ("wger.de Exercise Database",             fetch_wger_exercises),
    "meals":      ("TheMealDB Recipe Database",             fetch_themealdb_recipes),
    "meal_plans": ("Generated Meal Plan Templates",         generate_meal_plan_templates),
    "curated":    ("Curated Fitness Documents",             generate_curated_docs),
}


def dry_run():
    print("\n" + "=" * 65)
    print("  APEX AI Knowledge Base Builder v2 — DRY RUN")
    print("=" * 65)
    print(f"  Output: {RAW_DIR.resolve()}\n")
    for key, (name, _) in SOURCES.items():
        print(f"  [{key:12s}] {name}")
    print()


def build(sources: list[str] | None = None):
    print("\n" + "=" * 65)
    print("  APEX AI Knowledge Base Builder v2")
    print("=" * 65)
    print(f"  Output: {RAW_DIR.resolve()}\n")

    to_run = sources or list(SOURCES.keys())
    total = 0

    for key in to_run:
        if key not in SOURCES:
            log.warning(f"Unknown source: {key}"); continue
        name, fn = SOURCES[key]
        print(f"\n{'─'*65}")
        total += fn()

    print(f"\n{'='*65}")
    print(f"  ✅ Build complete! Files saved/verified: {total}")
    print(f"\n  Files in knowledge_base/raw/:")
    for f in sorted(RAW_DIR.glob("*")):
        if f.is_file() and f.suffix in (".txt", ".pdf", ".json"):
            print(f"    {f.name:50s} {f.stat().st_size//1024:>6} KB")
    print(f"\n  Next: POST /rag/reindex to rebuild FAISS index")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build APEX AI knowledge base")
    parser.add_argument("--source", nargs="+", choices=list(SOURCES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        build(sources=args.source)
