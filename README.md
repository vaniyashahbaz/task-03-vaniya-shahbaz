# task-03-vaniya-shahbaz
# AI Recommendation Engine


---

## Overview

A content-based recommendation system built in pure Python that matches user interest profiles to catalogue items using **cosine similarity**. No external libraries required — just the Python standard library.

This project demonstrates the foundational logic behind recommendation engines used in production at Netflix, Spotify, and YouTube — before neural networks enter the picture.

---

## How It Works

### The algorithm

Every user and item is represented as a **weighted feature vector**:

| Feature type | Weight |
|---|---|
| Selected category | 2.0 |
| Associated tag | 1.0 |

Similarity between a user profile and an item is computed using **cosine similarity**:

```
similarity = (U · I) / (‖U‖ × ‖I‖)
```

- `U · I` — dot product of both vectors (shared features)
- `‖U‖`, `‖I‖` — magnitudes (how rich each profile is)

The result is a score between **0** (no overlap) and **1** (perfect match). Only items at or above your chosen threshold are returned.

### Why cosine similarity?

It measures the *angle* between two vectors, not their size — so a user who selected one interest and an item with one matching tag score the same as a user with ten interests and an item with ten matching ones, as long as the direction aligns. This prevents popular/large items from always winning.

---

## Project Structure

```
project 3/
├── recommendation_system.py   ← main file (all logic + CLI)
└── README.md                  ← this file
```

---

## Getting Started

### Requirements

- Python 3.8 or higher
- No third-party packages needed

### Run the interactive CLI

```bash
cd "C:\Users\Momin\OneDrive\Desktop\internship_projects\project 3"
python recommendation_system.py
```

You'll be prompted to:
1. Pick interest categories from a numbered menu
2. Set a minimum match threshold (default: 20%)
3. Optionally limit how many results to show

### Example session

```
╔══════════════════════════════════════════════════╗
║   AI Recommendation Engine  —  DecodeLabs P3   ║
╚══════════════════════════════════════════════════╝

────────────────────────────────────────────────────
  Available interest categories:
────────────────────────────────────────────────────
   1. Technology
   2. Music
   3. Sports
   ...
────────────────────────────────────────────────────

Enter numbers (comma-separated, e.g. 1,3,9): 1,5
Minimum match threshold % (default 20): 30
Show top N results only? (leave blank for all): 5

════════════════════════════════════════════════════
  RECOMMENDATIONS
════════════════════════════════════════════════════
  Profile: Technology, Gaming
  Matched: 5 item(s)
────────────────────────────────────────────────────

  #1  Indie game dev in Unity
       [██████████████░░░░░░] 72.4%
       Categories : Gaming, Technology
       Tags       : indie, coding, software

  #2  VR game design principles
       [███████████░░░░░░░░░] 56.3%
       ...
```

---

## Use as a module

You can import the recommendation logic into any other Python file:

```python
from recommendation_system import get_recommendations

results = get_recommendations(
    selected_cats=["Technology", "Gaming"],
    threshold=0.3,
    top_n=5
)

for item in results:
    print(f"{item['score_pct']}% — {item['title']}")
```

### `get_recommendations()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `selected_cats` | `list[str]` | required | Category names the user selected |
| `threshold` | `float` | `0.2` | Minimum similarity score (0.0 – 1.0) |
| `top_n` | `int \| None` | `None` | Limit results to top N; `None` returns all |

### Return format

Each result is a dict:

```python
{
    "title":     "Indie game dev in Unity",
    "cats":      ["Gaming", "Technology"],
    "tags":      ["indie", "coding", "software"],
    "score":     0.7236,       # cosine similarity (0–1)
    "score_pct": 72.4          # same value as a percentage
}
```

---

## Available Categories

| # | Category | Sample tags |
|---|---|---|
| 1 | Technology | AI, coding, software, robotics |
| 2 | Music | guitar, jazz, piano, production |
| 3 | Sports | fitness, running, basketball |
| 4 | Science | physics, space, biology |
| 5 | Gaming | RPG, indie, VR, esports |
| 6 | Film & TV | drama, sci-fi, documentary |
| 7 | Travel | culture, adventure, photography |
| 8 | Art & Design | illustration, typography, UI |
| 9 | Books | fiction, history, philosophy |
| 10 | Food | cooking, vegan, street food |
| 11 | Health | meditation, sleep, nutrition |
| 12 | Business | startups, investing, finance |

---

## Key Concepts Demonstrated

- **Feature engineering** — converting raw preferences into weighted numeric vectors
- **Cosine similarity** — measuring directional alignment between vectors
- **Threshold filtering** — precision vs. recall tradeoff
- **Content-based filtering** — matching on item attributes, no user history needed
- **Modular design** — clean separation of data, algorithm, and interface layers

---

---

*DecodeLabs Internship · Project 3 · AI Recommendation Logic*
repository for task3 
