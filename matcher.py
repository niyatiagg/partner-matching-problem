from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CANONICAL_ALIASES: Dict[str, List[str]] = {
    "user_id": ["user_id", "id", "uid", "name", "user_name", "username"],
    "gender": ["gender", "sex"],
    "diet": ["diet", "dietary_restrictions", "food_preference"],
    "sleep_type": ["sleep_type", "sleeper_type"],
    "cleanliness": ["cleanliness", "hygiene", "cleaning_habit"],
    "noise_preference": ["noise_preference", "noise", "sound_preference"],
    "smoking_drinking": ["smoking_drinking", "smoking", "drinking"],
    "interests": ["interests", "hobbies", "tags", "topics"],
    "social_energy_rating": ["social_energy_rating", "social_energy", "extroversion_score"],
    "work_shift": ["work_shift", "shift", "work_time"],
    "profession": ["profession", "occupation", "job_role"],
    "room_type_preference": ["room_type_preference", "room_type"],
    "privacy_importance": ["privacy_importance", "privacy"],
    "working_style": ["working_style", "collab_style", "team_style"],
    "productive_time": ["productive_time", "peak_productivity", "best_work_time"],
    "priority_style": ["priority_style", "priority", "goal_style"],
    "time_commitment": ["time_commitment", "effort_level", "availability"],
    "goal_topic": ["goal_topic", "topic", "event_topic", "subject"],
    "domain": ["domain", "specialization", "track"],
    "meeting_preference": ["meeting_preference", "meeting_mode", "collab_mode"],
    "study_interests": ["study_interests", "study_topics", "subjects_of_interest"],
    "preferred_language": ["preferred_language", "language", "programming_language"],
    "duty_preference": ["duty_preference", "duty_pref", "house_duty_preference", "home_role"],
}


@dataclass
class MatchingConfig:
    n_users: int = 30
    edge_threshold: float = 0.45
    weight_cosine: float = 0.45
    weight_jaccard: float = 0.25
    weight_euclidean: float = 0.30
    output_dir: str = "outputs"
    region_registry_path: str = "regions.json"


def normalize_col(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def ensure_user_id(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()
    if "user_id" not in out.columns:
        out["user_id"] = [f"{region[:3].upper()}_{i+1:04d}" for i in range(len(out))]
    out["user_id"] = out["user_id"].astype(str)
    return out


def ensure_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()
    out["region"] = region
    return out


def build_synthetic_profiles(n_users: int, mode: str, region: str) -> pd.DataFrame:
    genders = ["female", "male", "other"]
    diets = ["vegetarian", "non-vegetarian", "vegan", "jain", "no restrictions"]
    sleep_types = ["light sleeper", "heavy sleeper", "mixed"]
    clean_options = ["messy", "both", "organized"]
    noise_options = ["quiet", "normal", "noisy"]
    smoke_options = ["non-smoker/non-drinker", "smoker", "drinker", "both", "okay with habits"]
    interests_pool = ["ai", "coding", "sports", "music", "travel", "movies", "reading", "gaming"]
    working_style_pool = ["independent", "collaborative", "balanced"]
    productive_time_pool = ["morning", "afternoon", "evening", "night"]
    priority_style_pool = ["grades", "innovation", "balanced"]
    commitment_pool = ["fast delivery", "deep effort", "balanced"]
    goal_topics = ["daa project", "hackathon ai", "college event", "startup idea"]
    language_pool = ["python", "java", "cpp", "javascript"]
    duty_pool = ["cooking", "cleaning"]

    records: List[Dict[str, object]] = []
    for i in range(n_users):
        interests = random.sample(interests_pool, k=random.randint(2, 5))
        rec: Dict[str, object] = {
            "user_id": f"{region[:3].upper()}_{i+1:04d}",
            "name": f"User_{i+1}",
            "region": region,
            "gender": random.choice(genders),
            "diet": random.choice(diets),
            "sleep_type": random.choice(sleep_types),
            "cleanliness": random.choice(clean_options),
            "noise_preference": random.choice(noise_options),
            "smoking_drinking": random.choice(smoke_options),
            "social_energy_rating": random.randint(1, 10),
            "interests": ",".join(interests),
            "openness": random.randint(1, 10),
            "conscientiousness": random.randint(1, 10),
            "extraversion": random.randint(1, 10),
            "agreeableness": random.randint(1, 10),
            "neuroticism": random.randint(1, 10),
            "working_style": random.choice(working_style_pool),
            "productive_time": random.choice(productive_time_pool),
            "priority_style": random.choice(priority_style_pool),
            "time_commitment": random.choice(commitment_pool),
            "goal_topic": random.choice(goal_topics),
            "preferred_language": random.choice(language_pool),
            "duty_preference": random.choice(duty_pool),
        }
        if mode == "roommate":
            rec["room_type_preference"] = random.choice(["private", "shared"])
            rec["privacy_importance"] = random.choice(["low", "medium", "high"])
        records.append(rec)
    return pd.DataFrame(records)


def best_alias_match(columns: List[str], aliases: List[str]) -> str | None:
    norm_to_orig = {normalize_col(c): c for c in columns}
    for alias in aliases:
        if alias in norm_to_orig:
            return norm_to_orig[alias]
    return None


def canonicalize_dataset(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]

    if "duty_pref" in out.columns and "duty_preference" in out.columns:
        good_pref = (
            out["duty_pref"].notna()
            & (out["duty_pref"].astype(str).str.strip() != "")
            & (out["duty_pref"].astype(str).str.strip().str.lower() != "nan")
        )
        out["duty_preference"] = out["duty_pref"].where(good_pref, out["duty_preference"])
        out = out.drop(columns=["duty_pref"])
    elif "duty_pref" in out.columns:
        out = out.rename(columns={"duty_pref": "duty_preference"})

    rename_map: Dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        found = best_alias_match(list(out.columns), aliases)
        if found:
            rename_map[found] = canonical
    out = out.rename(columns=rename_map)

    if "user_id" not in out.columns:
        if "user_name" in out.columns:
            out["user_id"] = out["user_name"]
        elif "name" in out.columns:
            out["user_id"] = out["name"]
        else:
            out["user_id"] = [f"{region[:3].upper()}_{i+1:04d}" for i in range(len(out))]
    out["user_id"] = out["user_id"].astype(str)

    if "interests" not in out.columns:
        out["interests"] = ""
    if "region" not in out.columns:
        out["region"] = region

    for col in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        if col not in out.columns:
            out[col] = 5

    # Best effort conversion to numeric where meaningful.
    for col in out.columns:
        if out[col].dtype == object:
            maybe_numeric = pd.to_numeric(out[col], errors="coerce")
            if maybe_numeric.notna().mean() > 0.8:
                out[col] = maybe_numeric

    return out


def infer_feature_columns(df: pd.DataFrame, mode: str) -> Tuple[List[str], List[str], List[str]]:
    exclude = {"user_id", "name", "region"}
    if mode == "roommate":
        # Scoring qualities only (hard filters are handled in app.py before scoring).
        preferred = {
            "social_energy_rating",
            "interests",
            "room_type_preference",
            "privacy_importance",
            "duty_preference",
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
    else:
        preferred = {
            "purpose",
            "meeting_preference",
            "domain",
            "work_preference",
            "study_interests",
            "preferred_language",
            "working_style",
            "productive_time",
            "priority_style",
            "time_commitment",
            "goal_topic",
            "interests",
            "social_energy_rating",
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }

    use_cols = [c for c in df.columns if c in preferred] or [c for c in df.columns if c not in exclude]
    # Only comma/semicolon indicate list-like fields; "/" alone matches labels like "Researcher/ Theoretical work".
    multi_cols = [
        c for c in use_cols if df[c].dtype == object and df[c].astype(str).str.contains(r",|;", regex=True).mean() > 0.4
    ]
    numeric_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in use_cols if c not in numeric_cols and c not in multi_cols]
    return numeric_cols, categorical_cols, multi_cols


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_weighted_graph(df: pd.DataFrame, cfg: MatchingConfig, mode: str) -> Tuple[nx.Graph, pd.DataFrame]:
    users = df["user_id"].astype(str).tolist()
    n = len(users)
    numeric_cols, categorical_cols, multi_cols = infer_feature_columns(df, mode)

    if numeric_cols:
        scaler = MinMaxScaler()
        numeric_scaled = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True)))
    else:
        numeric_scaled = np.zeros((n, 1))

    if categorical_cols:
        encoded_cat = pd.get_dummies(df[categorical_cols].fillna("unknown").astype(str), drop_first=False).values
    else:
        encoded_cat = np.zeros((n, 0))

    cosine_input = np.hstack([numeric_scaled, encoded_cat]) if encoded_cat.size else numeric_scaled
    cosine_mat = cosine_similarity(cosine_input)

    euclidean_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(numeric_scaled[i] - numeric_scaled[j])
            euclidean_mat[i, j] = 1.0 / (1.0 + dist)

    token_sets: List[set] = []
    for _, row in df.iterrows():
        s: set = set()
        for c in categorical_cols:
            s.add(f"{c}:{str(row[c]).strip().lower()}")
        for c in multi_cols:
            tokens = [x.strip().lower() for x in str(row[c]).replace(";", ",").replace("/", ",").split(",") if x.strip()]
            s.update({f"{c}:{tok}" for tok in tokens})
        token_sets.append(s)

    jaccard_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            jaccard_mat[i, j] = jaccard_similarity(token_sets[i], token_sets[j])

    combined = cfg.weight_cosine * cosine_mat + cfg.weight_jaccard * jaccard_mat + cfg.weight_euclidean * euclidean_mat

    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_node(
            str(row["user_id"]),
            label=str(row.get("name", row["user_id"])),
            region=str(row.get("region", "unknown")),
        )

    edge_rows: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(combined[i, j])
            edge_rows.append(
                {
                    "u": users[i],
                    "v": users[j],
                    "cosine": float(cosine_mat[i, j]),
                    "jaccard": float(jaccard_mat[i, j]),
                    "euclidean_similarity": float(euclidean_mat[i, j]),
                    "combined_weight": w,
                }
            )
            if w >= cfg.edge_threshold:
                graph.add_edge(users[i], users[j], weight=w)

    return graph, pd.DataFrame(edge_rows).sort_values("combined_weight", ascending=False)


def build_preference_lists_from_scores(scores: pd.DataFrame, users: List[str]) -> Dict[str, List[str]]:
    score_map: Dict[Tuple[str, str], float] = {}
    for _, row in scores.iterrows():
        u, v = str(row["u"]), str(row["v"])
        score_map[(u, v)] = float(row["combined_weight"])
        score_map[(v, u)] = float(row["combined_weight"])

    prefs: Dict[str, List[str]] = {}
    for u in users:
        others = [v for v in users if v != u]
        others.sort(key=lambda v: (score_map.get((u, v), 0.0), v), reverse=True)
        prefs[u] = others
    return prefs


def max_weight_roommate_matching(graph: nx.Graph) -> List[Tuple[str, str, float]]:
    matched = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    out = [(u, v, float(graph[u][v]["weight"])) for u, v in matched]
    return sorted(out, key=lambda x: x[2], reverse=True)


def gale_shapley_bipartite(preferences_a: Dict[str, List[str]], preferences_b: Dict[str, List[str]]) -> Dict[str, str]:
    free_a = list(preferences_a.keys())
    next_idx = {a: 0 for a in preferences_a}
    engaged_b: Dict[str, str] = {}
    rank_b = {b: {a: i for i, a in enumerate(lst)} for b, lst in preferences_b.items()}

    while free_a:
        a = free_a.pop(0)
        if next_idx[a] >= len(preferences_a[a]):
            continue
        b = preferences_a[a][next_idx[a]]
        next_idx[a] += 1
        if b not in engaged_b:
            engaged_b[b] = a
        else:
            curr = engaged_b[b]
            if rank_b[b][a] < rank_b[b][curr]:
                engaged_b[b] = a
                free_a.append(curr)
            else:
                free_a.append(a)
    return {a: b for b, a in engaged_b.items()}


def teammate_matching_via_gale_shapley(scores: pd.DataFrame, users: List[str], goal_topic: str | None) -> List[Tuple[str, str, float]]:
    shuffled = users[:]
    random.shuffle(shuffled)
    half = len(shuffled) // 2
    group_a = shuffled[:half]
    group_b = shuffled[half : half + half]
    score_map = {(str(r["u"]), str(r["v"])): float(r["combined_weight"]) for _, r in scores.iterrows()}
    score_map.update({(v, u): w for (u, v), w in list(score_map.items())})

    pref_a = {a: sorted(group_b, key=lambda b: (score_map.get((a, b), 0.0), b), reverse=True) for a in group_a}
    pref_b = {b: sorted(group_a, key=lambda a: (score_map.get((a, b), 0.0), a), reverse=True) for b in group_b}
    pairs = gale_shapley_bipartite(pref_a, pref_b)
    return sorted([(a, b, score_map.get((a, b), 0.0)) for a, b in pairs.items()], key=lambda x: x[2], reverse=True)


def stable_roommates_irving(preferences: Dict[str, List[str]]) -> Dict[str, str] | None:
    # Irving stable roommates algorithm (strict preferences).
    pref = {p: lst[:] for p, lst in preferences.items()}

    def remove_pair(a: str, b: str) -> None:
        if b in pref[a]:
            pref[a].remove(b)
        if a in pref[b]:
            pref[b].remove(a)

    # Phase 1: proposals and rejections.
    held_by: Dict[str, str] = {}
    next_choice = {p: 0 for p in pref}
    free = list(pref.keys())

    while free:
        p = free.pop(0)
        if next_choice[p] >= len(pref[p]):
            return None
        q = pref[p][next_choice[p]]
        next_choice[p] += 1

        if q not in held_by:
            held_by[q] = p
        else:
            current = held_by[q]
            q_pref = pref[q]
            if q_pref.index(p) < q_pref.index(current):
                held_by[q] = p
                free.append(current)
            else:
                free.append(p)
                continue

        proposer = held_by[q]
        cut_idx = pref[q].index(proposer)
        for r in pref[q][cut_idx + 1 :]:
            remove_pair(q, r)
            if len(pref[r]) == 0:
                return None
        if len(pref[q]) == 0:
            return None

    # Phase 2: eliminate rotations until all lists have length 1.
    while any(len(lst) > 1 for lst in pref.values()):
        p0 = next((p for p in pref if len(pref[p]) > 1), None)
        if p0 is None:
            break

        p_seq = [p0]
        q_seq: List[str] = []
        while True:
            p_last = p_seq[-1]
            q = pref[p_last][1]
            q_seq.append(q)
            p_next = pref[q][-1]
            if p_next in p_seq:
                start = p_seq.index(p_next)
                p_seq = p_seq[start:]
                q_seq = q_seq[start:]
                break
            p_seq.append(p_next)

        m = len(p_seq)
        for i in range(m):
            q_i = q_seq[i]
            p_next = p_seq[(i + 1) % m]
            remove_pair(q_i, p_next)
            if len(pref[q_i]) == 0 or len(pref[p_next]) == 0:
                return None

        # Trim dominated options after removals.
        changed = True
        while changed:
            changed = False
            for x in list(pref.keys()):
                if len(pref[x]) == 0:
                    return None
                if len(pref[x]) == 1:
                    y = pref[x][0]
                    idx = pref[y].index(x)
                    for z in pref[y][idx + 1 :]:
                        remove_pair(y, z)
                        changed = True
                        if len(pref[z]) == 0:
                            return None

    if any(len(lst) != 1 for lst in pref.values()):
        return None
    return {p: pref[p][0] for p in pref}


def map_pairs_to_scored_list(pair_map: Dict[str, str], scores: pd.DataFrame) -> List[Tuple[str, str, float]]:
    score_map = {(str(r["u"]), str(r["v"])): float(r["combined_weight"]) for _, r in scores.iterrows()}
    score_map.update({(v, u): w for (u, v), w in list(score_map.items())})
    seen = set()
    result: List[Tuple[str, str, float]] = []
    for u, v in pair_map.items():
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        result.append((u, v, score_map.get((u, v), 0.0)))
    return sorted(result, key=lambda x: x[2], reverse=True)


def export_interactive_graph(
    graph: nx.Graph,
    output_path: Path,
    highlight_pairs: List[Tuple[str, str, float]] | None = None,
    focus_user: str | None = None,
    max_nodes: int = 120,
    show_only_highlight_edges: bool = False,
    min_edge_weight: float = 0.0,
    max_edges_per_node: int = 0,
    show_labels: bool = True,
) -> None:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError("pyvis is required for visualization. Install requirements.txt.") from exc

    g = graph.copy()
    if focus_user and focus_user in g:
        nbrs = set(g.neighbors(focus_user))
        keep = nbrs | {focus_user}
        g = g.subgraph(keep).copy()
    elif g.number_of_nodes() > max_nodes:
        nodes_by_degree = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n, _ in nodes_by_degree}
        g = g.subgraph(keep).copy()

    pair_set = {tuple(sorted((a, b))) for a, b, _ in (highlight_pairs or [])}
    # Keep export-side pruning lightweight (node cap / focus only).
    # Edge granularity controls are injected into the output HTML and can be adjusted live without rerun.

    net = Network(height="780px", width="100%", bgcolor="#111111", font_color="white", cdn_resources="remote")
    net.force_atlas_2based()
    for node, attrs in g.nodes(data=True):
        color = "#EF5350" if focus_user and node == focus_user else "#4FC3F7"
        full_label = str(attrs.get("label", node))
        label = full_label if show_labels else ""
        net.add_node(
            node,
            label=label,
            full_label=full_label,
            title=f"{node}\nRegion: {attrs.get('region', '')}",
            color=color,
        )
    for u, v, data in g.edges(data=True):
        w = float(data.get("weight", 0.0))
        is_match = tuple(sorted((u, v))) in pair_set
        color = "#FFD54F" if is_match else "#90A4AE"
        net.add_edge(
            u,
            v,
            value=max(w * 8, 1.5),
            color=color,
            title=f"compatibility={w:.3f}",
            raw_weight=w,
            is_match=is_match,
        )
    net.save_graph(str(output_path))
    _inject_dynamic_graph_controls(
        output_path=output_path,
        defaults={
            "show_only_highlight_edges": bool(show_only_highlight_edges),
            "min_edge_weight": float(min_edge_weight),
            "max_edges_per_node": int(max_edges_per_node),
            "show_labels": bool(show_labels),
        },
    )


def _inject_dynamic_graph_controls(output_path: Path, defaults: Dict[str, object]) -> None:
    html = output_path.read_text(encoding="utf-8")
    if "id=\"ux-graph-controls\"" in html:
        return

    controls_html = """
<div id="ux-graph-controls" style="position:fixed; top:10px; right:10px; z-index:9999; background:#1f2937; color:#fff; padding:10px 12px; border-radius:8px; font-family:Arial,sans-serif; font-size:12px; width:250px; box-shadow:0 2px 8px rgba(0,0,0,0.35);">
  <div style="font-weight:700; margin-bottom:8px;">Graph Controls (live)</div>
  <label style="display:block; margin:6px 0;">
    <input id="ctl-match-only" type="checkbox" /> Show only matched edges
  </label>
  <label style="display:block; margin:6px 0;">Min edge weight: <span id="ctl-min-w-val"></span>
    <input id="ctl-min-w" type="range" min="0" max="1" step="0.01" style="width:100%;" />
  </label>
  <label style="display:block; margin:6px 0;">Max edges per node: <span id="ctl-k-val"></span>
    <input id="ctl-k" type="range" min="1" max="20" step="1" style="width:100%;" />
  </label>
  <label style="display:block; margin:6px 0;">
    <input id="ctl-labels" type="checkbox" /> Show labels
  </label>
  <div style="margin-top:8px; color:#d1d5db;">Applies instantly; no rerun needed.</div>
</div>
"""

    script = f"""
<script>
(function() {{
  const defaults = {json.dumps(defaults)};
  const controlsRoot = document.createElement("div");
  controlsRoot.innerHTML = `{controls_html}`;
  document.body.appendChild(controlsRoot.firstElementChild);

  if (typeof edges === "undefined" || typeof nodes === "undefined") {{
    return;
  }}

  const allEdges = edges.get().map((e, idx) => {{
    const key = e.id != null ? String(e.id) : `${{e.from}}__${{e.to}}__${{idx}}`;
    return {{...e, _edge_key:key, raw_weight: Number(e.raw_weight ?? 0), is_match: !!e.is_match }};
  }});
  const allNodes = nodes.get().map((n) => {{
    const full = n.full_label != null ? String(n.full_label) : String(n.label ?? n.id ?? "");
    return {{...n, _full_label: full }};
  }});

  const matchOnlyEl = document.getElementById("ctl-match-only");
  const minWEl = document.getElementById("ctl-min-w");
  const minWVal = document.getElementById("ctl-min-w-val");
  const kEl = document.getElementById("ctl-k");
  const kVal = document.getElementById("ctl-k-val");
  const labelsEl = document.getElementById("ctl-labels");

  matchOnlyEl.checked = !!defaults.show_only_highlight_edges;
  minWEl.value = Number(defaults.min_edge_weight || 0);
  kEl.value = Math.max(1, Number(defaults.max_edges_per_node || 5));
  labelsEl.checked = !!defaults.show_labels;

  function applyControls() {{
    const matchOnly = matchOnlyEl.checked;
    const minW = Number(minWEl.value);
    const k = Number(kEl.value);
    const showLabels = labelsEl.checked;
    minWVal.textContent = minW.toFixed(2);
    kVal.textContent = String(k);

    let candidate = allEdges.filter((e) => e.raw_weight >= minW && (!matchOnly || e.is_match));

    const byNode = new Map();
    candidate.forEach((e) => {{
      if (!byNode.has(e.from)) byNode.set(e.from, []);
      if (!byNode.has(e.to)) byNode.set(e.to, []);
      byNode.get(e.from).push(e);
      byNode.get(e.to).push(e);
    }});

    const keepKeys = new Set();
    byNode.forEach((arr) => {{
      arr.sort((a, b) => (b.raw_weight || 0) - (a.raw_weight || 0));
      arr.slice(0, k).forEach((e) => keepKeys.add(e._edge_key));
    }});
    const filtered = candidate.filter((e) => keepKeys.has(e._edge_key));

    edges.clear();
    edges.add(filtered);

    nodes.update(allNodes.map((n) => {{
      const next = {{...n}};
      next.label = showLabels ? n._full_label : "";
      return next;
    }}));
  }}

  [matchOnlyEl, minWEl, kEl, labelsEl].forEach((el) => el.addEventListener("input", applyControls));
  applyControls();
}})();
</script>
"""

    if "</body>" in html:
        html = html.replace("</body>", script + "\n</body>")
    else:
        html += script
    output_path.write_text(html, encoding="utf-8")


def prompt_choice(message: str, options: List[str]) -> str:
    print(f"\n{message}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Select option number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid input. Try again.")


def prompt_text(message: str, default: str = "") -> str:
    raw = input(f"{message}" + (f" [{default}]" if default else "") + ": ").strip()
    return raw if raw else default


def collect_user_profile(mode: str, existing_regions: List[str]) -> Dict[str, object]:
    region_choice = prompt_choice("Select your region", existing_regions + ["other"])
    if region_choice == "other":
        region_choice = prompt_text("Enter new region name", "new_region")

    profile: Dict[str, object] = {
        "user_id": prompt_text("Enter your user id (or leave empty to auto-generate)", ""),
        "name": prompt_text("Enter your name", "new_user"),
        "region": region_choice,
        "gender": prompt_choice("Gender?", ["female", "male", "other", "prefer_not_to_say"]),
        "diet": prompt_choice("Diet preference?", ["vegetarian", "non-vegetarian", "vegan", "jain", "no restrictions"]),
        "sleep_type": prompt_choice("What kind of sleeper are you?", ["light sleeper", "heavy sleeper", "mixed"]),
        "cleanliness": prompt_choice("Cleaning habit?", ["messy", "both", "organized"]),
        "noise_preference": prompt_choice("Noise preference?", ["quiet", "normal", "noisy"]),
        "smoking_drinking": prompt_choice(
            "Smoking / drinking preference?",
            ["non-smoker/non-drinker", "smoker", "drinker", "both", "okay with roommate habits"],
        ),
        "social_energy_rating": int(prompt_text("Social energy rating (1-10)", "5")),
        "interests": prompt_text("Interests (comma-separated)", "music,coding"),
        "openness": int(prompt_text("Openness (1-10)", "5")),
        "conscientiousness": int(prompt_text("Conscientiousness (1-10)", "5")),
        "extraversion": int(prompt_text("Extraversion (1-10)", "5")),
        "agreeableness": int(prompt_text("Agreeableness (1-10)", "5")),
        "neuroticism": int(prompt_text("Neuroticism (1-10)", "5")),
    }

    if mode == "roommate":
        profile["room_type_preference"] = prompt_choice("Room type preference?", ["private", "shared"])
        profile["privacy_importance"] = prompt_choice("Privacy importance?", ["low", "medium", "high"])
    else:
        profile["goal_topic"] = prompt_text(
            "Teammate purpose/topic (e.g. DAA project, Hackathon AI, Event analytics)",
            "daa project",
        )
        profile["working_style"] = prompt_choice("Working style?", ["independent", "collaborative", "balanced"])
        profile["productive_time"] = prompt_choice("Preferred working time?", ["morning", "afternoon", "evening", "night"])
        profile["priority_style"] = prompt_choice("More important?", ["grades", "innovation", "balanced"])
        profile["time_commitment"] = prompt_choice("Time commitment?", ["fast delivery", "deep effort", "balanced"])

    return profile


def apply_user_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    print("\nOptional filters (press Enter to skip):")
    for col in ["gender", "diet", "region"]:
        if col in out.columns:
            values = sorted(out[col].dropna().astype(str).unique().tolist())
            if values:
                print(f"\nAvailable {col} values: {values}")
                val = input(f"Filter by {col}? ").strip().lower()
                if val:
                    out = out[out[col].astype(str).str.lower() == val]
    return out.reset_index(drop=True)


def ensure_registry(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reg = json.load(f)
            # Backward-compatible migration for older registry files.
            changed = False
            for name, info in reg.items():
                if "kind" not in info:
                    info["kind"] = "teammate" if "class" in name.lower() else "roommate"
                    changed = True
            if "Computer Science - Class 2026" not in reg:
                reg["Computer Science - Class 2026"] = {
                    "type": "csv",
                    "path": "computer_science_class_2026.csv",
                    "kind": "teammate",
                }
                changed = True
            if "Hogwarts - Class 2026" not in reg:
                reg["Hogwarts - Class 2026"] = {"type": "csv", "path": "hogwarts_class_2026.csv", "kind": "teammate"}
                changed = True
            if "class1" in reg:
                del reg["class1"]
                changed = True
            if changed:
                with open(path, "w", encoding="utf-8") as fw:
                    json.dump(reg, fw, indent=2)
            return reg
    default = {
        "synthetic_region": {"type": "synthetic", "kind": "roommate"},
        "girls_pg_hostel_region": {"type": "csv", "path": "Girls_pg_hostel_CSV_data-1.csv", "kind": "roommate"},
        "Computer Science - Class 2026": {
            "type": "csv",
            "path": "computer_science_class_2026.csv",
            "kind": "teammate",
        },
        "Hogwarts - Class 2026": {"type": "csv", "path": "hogwarts_class_2026.csv", "kind": "teammate"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2)
    return default


def load_region_data(region: str, region_info: Dict[str, str], cfg: MatchingConfig, mode: str, base_dir: Path) -> pd.DataFrame:
    if region_info.get("type") == "synthetic":
        df = build_synthetic_profiles(cfg.n_users, mode=mode, region=region)
    else:
        csv_path = base_dir / region_info["path"]
        df = pd.read_csv(csv_path)
        df = canonicalize_dataset(df, region=region)
    df = ensure_user_id(df, region)
    df = ensure_region(df, region)
    return df


def run_mode(mode: str, cfg: MatchingConfig, region_registry: Dict[str, Dict[str, str]], base_dir: Path) -> None:
    region_names = sorted(region_registry.keys())
    selected_region = prompt_choice("Choose region to load data from", region_names)
    region_df = load_region_data(selected_region, region_registry[selected_region], cfg, mode, base_dir)

    add_self = prompt_choice("Do you want to add your own profile now?", ["yes", "no"]) == "yes"
    if add_self:
        profile = collect_user_profile(mode, region_names)
        if not profile.get("user_id"):
            profile["user_id"] = f"{profile['region'][:3].upper()}_SELF_{random.randint(100, 999)}"
        region_df = pd.concat([region_df, pd.DataFrame([profile])], ignore_index=True)

        if profile["region"] not in region_registry:
            region_registry[profile["region"]] = {"type": "synthetic"}

    filtered = apply_user_filters(region_df)
    if len(filtered) < 2:
        print("Not enough users after filtering. Please relax filters.")
        return

    graph, scores = build_weighted_graph(filtered, cfg, mode=mode)
    users = filtered["user_id"].astype(str).tolist()
    prefs = build_preference_lists_from_scores(scores, users)

    blossom_pairs = max_weight_roommate_matching(graph)
    gs_pairs = teammate_matching_via_gale_shapley(scores, users, goal_topic=None)
    irving_map = stable_roommates_irving(prefs)
    irving_pairs = map_pairs_to_scored_list(irving_map, scores) if irving_map else []

    out_dir = base_dir / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_dir / f"{mode}_profiles_used.csv", index=False)
    scores.to_csv(out_dir / f"{mode}_pair_scores.csv", index=False)

    results_payload = {
        "mode": mode,
        "region": selected_region,
        "blossom_max_weight_matching": [{"user_a": a, "user_b": b, "score": s} for a, b, s in blossom_pairs],
        "gale_shapley_matching": [{"user_a": a, "user_b": b, "score": s} for a, b, s in gs_pairs],
        "irving_stable_roommates": [{"user_a": a, "user_b": b, "score": s} for a, b, s in irving_pairs],
        "irving_status": "stable matching found" if irving_pairs else "no stable matching found",
    }
    with open(out_dir / f"{mode}_matching_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    focus_user = None
    if add_self and "user_id" in profile:
        focus_user = str(profile["user_id"])
    export_interactive_graph(
        graph,
        out_dir / f"{mode}_compatibility_graph.html",
        highlight_pairs=blossom_pairs if mode == "roommate" else gs_pairs,
        focus_user=focus_user,
    )

    print("\nRun complete.")
    print(f"Mode: {mode}")
    print(f"Region: {selected_region}")
    print(f"Users considered: {len(filtered)}")
    print(f"Edges above threshold: {graph.number_of_edges()}")
    print(f"Blossom pairs: {len(blossom_pairs)}")
    print(f"Gale-Shapley pairs: {len(gs_pairs)}")
    print(f"Irving stable roommates pairs: {len(irving_pairs)}")
    print(f"Outputs saved in: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flexible graph matcher: roommate + teammate")
    parser.add_argument("--n_users", type=int, default=30, help="Synthetic users per region.")
    parser.add_argument("--edge_threshold", type=float, default=0.45, help="Minimum edge weight.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--weight_cosine", type=float, default=0.45, help="Cosine contribution.")
    parser.add_argument("--weight_jaccard", type=float, default=0.25, help="Jaccard contribution.")
    parser.add_argument("--weight_euclidean", type=float, default=0.30, help="Euclidean contribution.")
    parser.add_argument("--region_registry_path", type=str, default="regions.json", help="JSON file with region data sources.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MatchingConfig(
        n_users=args.n_users,
        edge_threshold=args.edge_threshold,
        weight_cosine=args.weight_cosine,
        weight_jaccard=args.weight_jaccard,
        weight_euclidean=args.weight_euclidean,
        output_dir=args.output_dir,
        region_registry_path=args.region_registry_path,
    )
    base_dir = Path(__file__).resolve().parent
    registry_path = base_dir / cfg.region_registry_path
    region_registry = ensure_registry(registry_path)

    mode = prompt_choice("What do you want to do?", ["find roommate", "find teammate"])
    canonical_mode = "roommate" if mode == "find roommate" else "teammate"
    run_mode(canonical_mode, cfg, region_registry, base_dir)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(region_registry, f, indent=2)


if __name__ == "__main__":
    main()
