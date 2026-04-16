# Graph-Based Roommate/Teammate Matcher (DAA Project)

This project models people as graph nodes and compatibility as weighted edges.
It supports:

- **Roommate mode** using **Maximum Weight Matching (Blossom via NetworkX)**.
- **Teammate mode** using **Gale-Shapley Stable Matching** on bipartite splits.
- Edge-weight construction using your professor's suggested metrics:
  - Jaccard similarity
  - Cosine similarity
  - Euclidean-distance-based similarity

## Why this algorithm choice?

- **Primary for roommate finder:** Maximum Weight Matching is the most direct fit when your objective is to maximize total compatibility.
- **Primary for teammate finder:** Gale-Shapley is a strong option when two-sided preferences are natural (Group A and Group B).
- **Irving's Stable Roommates** is a valid advanced extension for strict stability in one-sided roommate markets, but it is more complex to implement/debug in a semester timeline.

## Project Structure

- `matcher.py`: end-to-end implementation.
- `requirements.txt`: Python dependencies.
- `outputs/` (generated after run):
  - `profiles_used.csv`
  - `pair_scores.csv`
  - `roommate_matching.json`
  - `teammate_matching.json`
  - `compatibility_graph.html` (interactive)

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run (Synthetic Data)

```bash
python matcher.py --n_users 20 --edge_threshold 0.45
```

## Run (Your Own CSV)

```bash
python matcher.py --csv_path your_profiles.csv --edge_threshold 0.45
```

Expected CSV columns:

- `user_id`
- `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism` (1-10)
- `sleep_schedule`
- `cleanliness`
- `diet`
- `social_style`
- `study_style`
- `interests` (comma-separated, example: `music,travel,gaming`)

## Suggested Team Work Split

### Partner 1 (Algorithm + Analysis)
- Implement and tune weighting model.
- Run complexity/time comparison across user counts.
- Evaluate stability and quality metrics.

### Partner 2 (Data + Visualization + Report)
- Build/clean dataset (Kaggle + synthetic).
- Build interactive graph UI outputs.
- Write experimental results, charts, and conclusion.

## Recommended Evaluation for Final Demo

1. Compare weight-computation variants:
   - cosine only
   - cosine + jaccard
   - cosine + jaccard + euclidean
2. Compare matching quality:
   - total compatibility score
   - average pair score
3. Compare runtime with increasing `n` (20, 40, 80, 160).
4. Visual demo in `compatibility_graph.html`.
