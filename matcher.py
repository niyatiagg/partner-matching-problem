from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class MatchingConfig:
    n_users: int = 20
    edge_threshold: float = 0.45
    weight_cosine: float = 0.45
    weight_jaccard: float = 0.25
    weight_euclidean: float = 0.30
    output_dir: str = "outputs"


def build_synthetic_profiles(n_users: int) -> pd.DataFrame:
    sleep_options = ["early_bird", "night_owl"]
    clean_options = ["low", "medium", "high"]
    diet_options = ["omnivore", "vegetarian", "vegan"]
    social_options = ["introvert", "ambivert", "extrovert"]
    study_options = ["quiet", "music", "group"]

    interest_pool = [
        "gym",
        "gaming",
        "hiking",
        "cooking",
        "movies",
        "travel",
        "reading",
        "music",
        "sports",
        "art",
    ]

    records: List[Dict[str, object]] = []
    for i in range(n_users):
        selected_interests = random.sample(interest_pool, k=random.randint(2, 5))
        records.append(
            {
                "user_id": f"U{i+1:02d}",
                "openness": random.randint(1, 10),
                "conscientiousness": random.randint(1, 10),
                "extraversion": random.randint(1, 10),
                "agreeableness": random.randint(1, 10),
                "neuroticism": random.randint(1, 10),
                "sleep_schedule": random.choice(sleep_options),
                "cleanliness": random.choice(clean_options),
                "diet": random.choice(diet_options),
                "social_style": random.choice(social_options),
                "study_style": random.choice(study_options),
                "interests": ",".join(sorted(selected_interests)),
            }
        )
    return pd.DataFrame(records)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_weighted_graph(df: pd.DataFrame, cfg: MatchingConfig) -> Tuple[nx.Graph, pd.DataFrame]:
    numeric_cols = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    categorical_cols = ["sleep_schedule", "cleanliness", "diet", "social_style", "study_style"]

    users = df["user_id"].tolist()
    n = len(users)

    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])

    encoded_cat = pd.get_dummies(df[categorical_cols], drop_first=False).values
    cosine_input = np.hstack([numeric_scaled, encoded_cat])
    cosine_mat = cosine_similarity(cosine_input)

    # Euclidean similarity on normalized numeric traits.
    euclidean_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            distance = np.linalg.norm(numeric_scaled[i] - numeric_scaled[j])
            euclidean_mat[i, j] = 1.0 / (1.0 + distance)

    # Jaccard over interests + key lifestyle tags.
    lifestyle_sets = []
    for _, row in df.iterrows():
        s = {
            f"sleep:{row['sleep_schedule']}",
            f"clean:{row['cleanliness']}",
            f"diet:{row['diet']}",
            f"social:{row['social_style']}",
            f"study:{row['study_style']}",
        }
        interest_tokens = [x.strip() for x in str(row["interests"]).split(",") if x.strip()]
        s.update({f"interest:{x}" for x in interest_tokens})
        lifestyle_sets.append(s)

    jaccard_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            jaccard_mat[i, j] = jaccard_similarity(lifestyle_sets[i], lifestyle_sets[j])

    combined = (
        cfg.weight_cosine * cosine_mat
        + cfg.weight_jaccard * jaccard_mat
        + cfg.weight_euclidean * euclidean_mat
    )

    graph = nx.Graph()
    for uid in users:
        graph.add_node(uid)

    edge_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(combined[i, j])
            if score >= cfg.edge_threshold:
                graph.add_edge(users[i], users[j], weight=score)
            edge_rows.append(
                {
                    "u": users[i],
                    "v": users[j],
                    "cosine": float(cosine_mat[i, j]),
                    "jaccard": float(jaccard_mat[i, j]),
                    "euclidean_similarity": float(euclidean_mat[i, j]),
                    "combined_weight": score,
                }
            )

    edge_scores_df = pd.DataFrame(edge_rows).sort_values("combined_weight", ascending=False)
    return graph, edge_scores_df


def max_weight_roommate_matching(graph: nx.Graph) -> List[Tuple[str, str, float]]:
    matched_edges = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    result = []
    for u, v in matched_edges:
        result.append((u, v, float(graph[u][v]["weight"])))
    return sorted(result, key=lambda x: x[2], reverse=True)


def gale_shapley_bipartite(
    preferences_a: Dict[str, List[str]], preferences_b: Dict[str, List[str]]
) -> Dict[str, str]:
    """Compute a stable matching for two disjoint sets using Gale-Shapley."""
    group_a = list(preferences_a.keys())
    group_b = list(preferences_b.keys())
    set_a = set(group_a)
    set_b = set(group_b)

    # Keep only valid counterparts and remove duplicates while preserving order.
    sanitized_a: Dict[str, List[str]] = {}
    for a, pref in preferences_a.items():
        seen: set[str] = set()
        cleaned = []
        for b in pref:
            if b in set_b and b not in seen:
                cleaned.append(b)
                seen.add(b)
        sanitized_a[a] = cleaned

    rank_b: Dict[str, Dict[str, int]] = {}
    for b, pref in preferences_b.items():
        seen: set[str] = set()
        cleaned = []
        for a in pref:
            if a in set_a and a not in seen:
                cleaned.append(a)
                seen.add(a)
        ranking = {a: i for i, a in enumerate(cleaned)}
        # Unlisted proposers are least preferred, but still comparable.
        fallback_rank = len(cleaned)
        for a in group_a:
            ranking.setdefault(a, fallback_rank)
        rank_b[b] = ranking

    free_a = deque(group_a)
    next_proposal_index = {a: 0 for a in group_a}
    engagement_b: Dict[str, str] = {}

    while free_a:
        a = free_a.popleft()
        pref_list = sanitized_a.get(a, [])
        if next_proposal_index[a] >= len(pref_list):
            continue

        b = pref_list[next_proposal_index[a]]
        next_proposal_index[a] += 1

        current = engagement_b.get(b)
        if current is None:
            engagement_b[b] = a
            continue

        if rank_b[b][a] < rank_b[b][current]:
            engagement_b[b] = a
            free_a.append(current)
        else:
            free_a.append(a)

    return {a: b for b, a in engagement_b.items()}


def teammate_matching_via_stable_marriage(graph: nx.Graph, users: List[str]) -> List[Tuple[str, str, float]]:
    shuffled = users[:]
    random.shuffle(shuffled)
    mid = len(shuffled) // 2
    group_a = shuffled[:mid]
    group_b = shuffled[mid:]

    # Keep bipartite sides equal-sized for stable marriage assumptions.
    if len(group_b) > len(group_a):
        group_b = group_b[: len(group_a)]
    elif len(group_a) > len(group_b):
        group_a = group_a[: len(group_b)]

    preferences_a = {}
    for a in group_a:
        sorted_b = sorted(group_b, key=lambda b: graph[a][b]["weight"] if graph.has_edge(a, b) else 0.0, reverse=True)
        preferences_a[a] = sorted_b

    preferences_b = {}
    for b in group_b:
        sorted_a = sorted(group_a, key=lambda a: graph[a][b]["weight"] if graph.has_edge(a, b) else 0.0, reverse=True)
        preferences_b[b] = sorted_a

    matches = gale_shapley_bipartite(preferences_a, preferences_b)
    result = []
    for a, b in matches.items():
        score = float(graph[a][b]["weight"]) if graph.has_edge(a, b) else 0.0
        result.append((a, b, score))
    return sorted(result, key=lambda x: x[2], reverse=True)


def export_interactive_graph(graph: nx.Graph, output_path: Path) -> None:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError(
            "pyvis is required for interactive visualization. Install dependencies from requirements.txt."
        ) from exc

    net = Network(height="750px", width="100%", bgcolor="#111111", font_color="white")
    net.force_atlas_2based()

    for node in graph.nodes:
        net.add_node(node, label=node, color="#4FC3F7")

    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 0.0)
        net.add_edge(u, v, value=w * 8, title=f"weight={w:.3f}")

    net.save_graph(str(output_path))


def run_pipeline(cfg: MatchingConfig, csv_path: str | None) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = build_synthetic_profiles(cfg.n_users)

    graph, edge_df = build_weighted_graph(df, cfg)

    room_matches = max_weight_roommate_matching(graph)
    team_matches = teammate_matching_via_stable_marriage(graph, df["user_id"].tolist())

    df.to_csv(out_dir / "profiles_used.csv", index=False)
    edge_df.to_csv(out_dir / "pair_scores.csv", index=False)

    with open(out_dir / "roommate_matching.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"user_a": a, "user_b": b, "score": s} for a, b, s in room_matches],
            f,
            indent=2,
        )

    with open(out_dir / "teammate_matching.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"user_a": a, "user_b": b, "score": s} for a, b, s in team_matches],
            f,
            indent=2,
        )

    export_interactive_graph(graph, out_dir / "compatibility_graph.html")

    print("Run complete.")
    print(f"Users: {len(df)}")
    print(f"Graph edges above threshold: {graph.number_of_edges()}")
    print(f"Roommate pairs (max weight): {len(room_matches)}")
    print(f"Teammate pairs (Gale-Shapley): {len(team_matches)}")
    if room_matches:
        print("Top roommate pair:", room_matches[0])


def parse_args() -> MatchingConfig:
    parser = argparse.ArgumentParser(description="Graph-based matcher for roommates/teammates.")
    parser.add_argument("--n_users", type=int, default=20, help="Number of synthetic users.")
    parser.add_argument("--edge_threshold", type=float, default=0.45, help="Minimum edge weight.")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional path to user profile CSV.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--weight_cosine", type=float, default=0.45, help="Cosine contribution.")
    parser.add_argument("--weight_jaccard", type=float, default=0.25, help="Jaccard contribution.")
    parser.add_argument("--weight_euclidean", type=float, default=0.30, help="Euclidean contribution.")
    args = parser.parse_args()

    cfg = MatchingConfig(
        n_users=args.n_users,
        edge_threshold=args.edge_threshold,
        weight_cosine=args.weight_cosine,
        weight_jaccard=args.weight_jaccard,
        weight_euclidean=args.weight_euclidean,
        output_dir=args.output_dir,
    )
    run_pipeline(cfg, args.csv_path)
    return cfg


if __name__ == "__main__":
    parse_args()
