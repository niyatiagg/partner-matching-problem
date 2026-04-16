# Graph-Based Roommate / Teammate Matching (DAA Project)

Finding compatible roommates (or teammates) is hard because compatibility depends on many lifestyle and personality factors. We model each person as a **vertex** in a graph and pairwise compatibility as **weighted edges**, then apply graph matching algorithms to propose pairs.

## Key idea

- Build a **compatibility score** from user attributes (OCEAN traits, sleep, diet, cleanliness, lifestyle, interests).
- Use scores as **edge weights** (maximum-weight matching) or derive **preference lists** (Gale–Shapley).
- Compare approaches on total compatibility, stability (where defined), and runtime.

## Algorithms (proposal + current code)

| Approach | Role in this repo |
|----------|-------------------|
| **Edmonds / Blossom — maximum weight matching** | **Roommate mode** — maximize total compatibility (`networkx.max_weight_matching`). |
| **Gale–Shapley stable matching** | **Teammate mode** — bipartite split + stable pairs. |
| **Irving stable roommates** | Planned extension (general roommate stability). |

## Current prototype (implemented)

- **`matcher.py`** — synthetic or CSV profiles → cosine + Jaccard + Euclidean similarity → weighted graph (threshold) → matchings → **PyVis** HTML graph.
- **`requirements.txt`** — Python dependencies.
- **`Sprint_1_Progress_Report.md`** — sprint submission notes.
- **`outputs/`** — generated when you run the script (gitignored): CSVs, JSON matchings, `compatibility_graph.html`.

### Setup

```bash
python -m pip install -r requirements.txt
```

### Run (synthetic data)

```bash
python matcher.py --n_users 20 --edge_threshold 0.45
```

### Run (your CSV)

```bash
python matcher.py --csv_path your_profiles.csv --edge_threshold 0.45
```

**Expected CSV columns:** `user_id`, `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`, `sleep_schedule`, `cleanliness`, `diet`, `social_style`, `study_style`, `interests` (comma-separated).

## Scoring (high level)

Combined edge weight uses tunable contributions: cosine (numeric + encoded categoricals), Jaccard (interests + lifestyle tags), and Euclidean-based similarity on OCEAN (see `matcher.py` for formulas and defaults).

## Evaluation plan (semester)

- Stability / blocking pairs where applicable; total and average compatibility; runtime vs. \(n\); sensitivity to weights and threshold.
- Datasets: public profiles (e.g. Kaggle) **mapped** to this schema, plus synthetic stress tests.

## Deliverables (target)

Working pipeline, algorithm comparison, visualizations, slides + short report.

**Repository:** [github.com/niyatiagg/partner-matching-problem](https://github.com/niyatiagg/partner-matching-problem)
