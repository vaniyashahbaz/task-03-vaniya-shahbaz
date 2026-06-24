"""
AI Recommendation System — DecodeLabs Project 3
Pattern Alignment via Cosine Similarity
"""

import math
from typing import Optional


#  DATA

CATEGORIES: dict[str, list[str]] = {
    "Technology":   ["AI", "coding", "gadgets", "hardware", "software", "robotics", "cloud"],
    "Music":        ["concerts", "guitar", "piano", "jazz", "pop", "hip-hop", "production"],
    "Sports":       ["fitness", "football", "basketball", "running", "cycling", "swimming"],
    "Science":      ["physics", "biology", "space", "chemistry", "research", "nature"],
    "Gaming":       ["RPG", "strategy", "indie", "multiplayer", "VR", "esports"],
    "Film & TV":    ["drama", "comedy", "thriller", "animation", "documentary", "sci-fi"],
    "Travel":       ["culture", "adventure", "food", "photography", "history", "nature"],
    "Art & Design": ["illustration", "typography", "photography", "sculpture", "UI", "fashion"],
    "Books":        ["fiction", "history", "philosophy", "science", "biography", "self-help"],
    "Food":         ["cooking", "baking", "vegan", "world cuisine", "wine", "street food"],
    "Health":       ["yoga", "meditation", "nutrition", "sleep", "mental health", "hiking"],
    "Business":     ["startups", "investing", "marketing", "leadership", "finance", "economics"],
}

ITEMS: list[dict] = [
    {"title": "Neural networks explained",      "cats": ["Technology", "Science"],              "tags": ["AI", "research", "software"]},
    {"title": "Guitar for beginners",           "cats": ["Music"],                              "tags": ["guitar", "indie", "concerts"]},
    {"title": "Marathon training plan",         "cats": ["Sports", "Health"],                  "tags": ["running", "fitness", "nutrition"]},
    {"title": "The cosmos: a visual guide",     "cats": ["Science", "Books"],                  "tags": ["space", "nature", "history"]},
    {"title": "Elden Ring deep dive",           "cats": ["Gaming", "Film & TV"],               "tags": ["RPG", "fantasy", "strategy"]},
    {"title": "Tokyo travel essentials",        "cats": ["Travel", "Food"],                    "tags": ["culture", "food", "photography"]},
    {"title": "Generative AI art tools",        "cats": ["Technology", "Art & Design"],        "tags": ["AI", "illustration", "software"]},
    {"title": "History of jazz",                "cats": ["Music", "Books"],                    "tags": ["jazz", "history", "biography"]},
    {"title": "Plant-based cooking master",     "cats": ["Food", "Health"],                    "tags": ["vegan", "cooking", "nutrition"]},
    {"title": "VC funding & startup law",       "cats": ["Business", "Technology"],            "tags": ["startups", "finance", "coding"]},
    {"title": "Indie game dev in Unity",        "cats": ["Gaming", "Technology"],              "tags": ["indie", "coding", "software"]},
    {"title": "Documentary filmmaking",         "cats": ["Film & TV", "Art & Design"],         "tags": ["documentary", "photography", "UI"]},
    {"title": "Rock climbing fundamentals",     "cats": ["Sports", "Travel"],                  "tags": ["adventure", "fitness", "nature"]},
    {"title": "Typography masterclass",         "cats": ["Art & Design", "Books"],             "tags": ["typography", "UI", "self-help"]},
    {"title": "Quantum physics primer",         "cats": ["Science", "Technology"],             "tags": ["physics", "research", "AI"]},
    {"title": "Piano chord progressions",       "cats": ["Music", "Art & Design"],             "tags": ["piano", "pop", "production"]},
    {"title": "Mindfulness & sleep science",    "cats": ["Health", "Science"],                 "tags": ["meditation", "sleep", "biology"]},
    {"title": "Formula 1 strategy guide",       "cats": ["Sports", "Business"],               "tags": ["strategy", "finance", "leadership"]},
    {"title": "Bouldering around the world",    "cats": ["Travel", "Sports"],                  "tags": ["adventure", "culture", "fitness"]},
    {"title": "Sci-fi cinema deep cuts",        "cats": ["Film & TV", "Books"],               "tags": ["sci-fi", "fiction", "drama"]},
    {"title": "Personal finance for 30s",       "cats": ["Business", "Health"],               "tags": ["investing", "mental health", "economics"]},
    {"title": "Street food photography",        "cats": ["Food", "Art & Design", "Travel"],   "tags": ["photography", "street food", "culture"]},
    {"title": "VR game design principles",      "cats": ["Gaming", "Technology", "Art & Design"], "tags": ["VR", "UI", "software"]},
    {"title": "Fermentation & gut health",      "cats": ["Food", "Health", "Science"],        "tags": ["cooking", "biology", "nutrition"]},
    {"title": "Esports business models",        "cats": ["Gaming", "Business"],               "tags": ["esports", "marketing", "finance"]},
]


#
#  CORE ALGORITHM
#

def build_user_vector(selected_cats: list[str]) -> dict[str, float]:
    """
    Build a weighted feature vector from the user's selected categories.
    Categories get weight 2; their associated tags get weight 1.
    """
    vec: dict[str, float] = {}
    for cat in selected_cats:
        if cat not in CATEGORIES:
            continue
        vec[cat] = vec.get(cat, 0) + 2.0
        for tag in CATEGORIES[cat]:
            vec[tag] = vec.get(tag, 0) + 1.0
    return vec


def build_item_vector(item: dict) -> dict[str, float]:
    """
    Build a weighted feature vector for a catalogue item.
    Same weighting scheme: categories × 2, tags × 1.
    """
    vec: dict[str, float] = {}
    for cat in item["cats"]:
        vec[cat] = vec.get(cat, 0) + 2.0
    for tag in item["tags"]:
        vec[tag] = vec.get(tag, 0) + 1.0
    return vec


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Cosine similarity = (A · B) / (||A|| × ||B||)
    Returns a value in [0, 1]. Returns 0 if either vector is zero.
    """
    all_keys = set(vec_a) | set(vec_b)
    dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def get_recommendations(
    selected_cats: list[str],
    threshold: float = 0.2,
    top_n: Optional[int] = None,
) -> list[dict]:
    """
    Return items ranked by cosine similarity to the user's profile.

    Args:
        selected_cats: Category names the user selected.
        threshold:     Minimum similarity score (0–1) to include an item.
        top_n:         If set, return only the top N results.

    Returns:
        List of dicts with item data + 'score' and 'score_pct' fields,
        sorted by score descending.
    """
    user_vec = build_user_vector(selected_cats)
    results = []
    for item in ITEMS:
        item_vec = build_item_vector(item)
        score = cosine_similarity(user_vec, item_vec)
        if score >= threshold:
            results.append({**item, "score": round(score, 4), "score_pct": round(score * 100, 1)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n] if top_n else results


# ─────────────────────────────────────────────
#  CLI INTERFACE
# ─────────────────────────────────────────────

def print_menu() -> None:
    print("\n" + "─" * 52)
    print("  Available interest categories:")
    print("─" * 52)
    cats = list(CATEGORIES.keys())
    for i, cat in enumerate(cats, 1):
        print(f"  {i:>2}. {cat}")
    print("─" * 52)


def get_user_selections() -> list[str]:
    cats = list(CATEGORIES.keys())
    print_menu()
    raw = input("\nEnter numbers (comma-separated, e.g. 1,3,9): ").strip()
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(cats):
                selected.append(cats[idx])
            else:
                print(f"  ⚠ Skipping out-of-range number: {part}")
        elif part:
            print(f"  ⚠ Skipping invalid input: '{part}'")
    return selected


def get_threshold() -> float:
    raw = input("Minimum match threshold % (default 20): ").strip()
    if not raw:
        return 0.20
    try:
        val = float(raw)
        return max(0.0, min(val / 100, 1.0))
    except ValueError:
        print("  Invalid input — using default 20%.")
        return 0.20


def display_results(results: list[dict], selected: list[str]) -> None:
    print("\n" + "═" * 52)
    print("  RECOMMENDATIONS")
    print("═" * 52)
    print(f"  Profile: {', '.join(selected)}")
    print(f"  Matched: {len(results)} item(s)")
    print("─" * 52)
    if not results:
        print("  No items meet your threshold.")
        print("  Try lowering it or selecting more interests.")
    else:
        for rank, item in enumerate(results, 1):
            bar_len = int(item["score"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"\n  #{rank}  {item['title']}")
            print(f"       [{bar}] {item['score_pct']}%")
            print(f"       Categories : {', '.join(item['cats'])}")
            print(f"       Tags       : {', '.join(item['tags'])}")
    print("═" * 52)


def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   AI Recommendation Engine    ║")
    print("╚══════════════════════════════════════════════════╝")

    while True:
        selected = get_user_selections()
        if not selected:
            print("  No valid categories selected. Please try again.")
            continue

        threshold = get_threshold()
        top_n_raw = input("Show top N results only? (leave blank for all): ").strip()
        top_n = int(top_n_raw) if top_n_raw.isdigit() else None

        results = get_recommendations(selected, threshold=threshold, top_n=top_n)
        display_results(results, selected)

        again = input("\n  Run again? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()