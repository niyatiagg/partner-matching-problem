# Graph Roommate / Teammate Matcher (DAA project)

This is a full interactive project prototype:

- Supports **find roommate** and **find teammate** modes.
- Supports **region-based datasets** (synthetic + CSV regions).
- Lets user answer profile questions, apply hard filters, run matching, and visualize graph.
- Compares multiple matching algorithms in the same run.

## Files

| Item | Role |
|------|------|
| `matcher.py` | Main app (interactive CLI + graph + matching + exports) |
| `Girls_pg_hostel_CSV_data-1.csv` | Real roommate-style region dataset |
| `regions.json` | Auto-created region registry (`region -> source`) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated outputs (profiles, scores, JSON results, graph HTML) |

## Major features implemented

1. **Dataset-friendly ingestion (less hardcoding)**
   - CSV headers are normalized automatically.
   - Alias-based canonical mapping handles common column-name variations.
   - If OCEAN columns are missing, fallback defaults are added so pipeline still runs.
   - Works directly with your hostel CSV.

2. **Interactive user profile questionnaire**
   - Asks mode first: `find roommate` or `find teammate`.
   - Asks region (including `other` to create a new region).
   - Asks profile questions (sleep type, cleanliness, diet, etc.).
   - Teammate mode asks purpose/topic + work preferences.

3. **Filtering**
   - Optional strict filters by available fields (gender/diet/region where present).
   - Reduces candidate pool before matching.

4. **Similarity scoring**
   - `Cosine similarity` on numeric + one-hot categorical features.
   - `Euclidean-distance based similarity` on normalized numeric features.
   - `Jaccard similarity` on tokenized categorical + multi-value text fields.
   - Final edge weight:
     - `0.45 * cosine + 0.25 * jaccard + 0.30 * euclidean_similarity` (tunable flags).

5. **Graph matching algorithms**
   - **Blossom maximum-weight matching** (roommate objective: maximize total compatibility).
   - **Gale-Shapley stable matching** (teammate objective on bipartite split).
   - **Irving stable roommates** (one-sided stable roommate matching attempt; may report no stable solution for some instances).

6. **Visualization**
   - Interactive PyVis graph HTML output.
   - Supports highlighting selected matched edges.
   - Supports focused view around current user or top-degree subset if graph is huge.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Web UI (recommended)

```bash
streamlit run app.py
```

Then open the localhost URL shown in terminal (typically `http://localhost:8501`).

## Run

```bash
python matcher.py --n_users 30 --edge_threshold 0.45
```

You will be prompted step-by-step in terminal.

## Output files

Under `outputs/`:

- `<mode>_profiles_used.csv`
- `<mode>_pair_scores.csv`
- `<mode>_matching_results.json`
- `<mode>_compatibility_graph.html`

For UI runs:

- `<mode>_matching_results_ui.json`
- `<mode>_compatibility_graph_ui.html`

## Data storage (current architecture)

- No database is used right now.
- Region datasets are persisted as CSV files in `data/regions/` (one CSV per region).
- Region metadata is tracked in `regions.json`.
- Generated algorithm outputs and graph files are saved in `outputs/`.
- Only `synthetic_region` auto-generates synthetic users. New custom regions remain empty until users are added.
