from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from matcher import (
    CANONICAL_ALIASES,
    MatchingConfig,
    build_preference_lists_from_scores,
    build_weighted_graph,
    canonicalize_dataset,
    ensure_registry,
    export_interactive_graph,
    gale_shapley_bipartite,
    map_pairs_to_scored_list,
    max_weight_roommate_matching,
    stable_roommates_irving,
)

# OCEAN + study-interest catalog (teammate self-service & schema)
OCEAN_COLS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
STUDY_INTERESTS_13 = [
    "AI-ML",
    "Cybersecurity",
    "Blockchain",
    "Software Engineering",
    "Data Science & Data Mining",
    "Computer Architecture & Hardware",
    "Networking & Distributed Systems",
    "Computer Graphics & Vizualization",
    "Human-Computer Interaction",
    "Robotics",
    "Theory of Computation",
    "Cloud Computing",
    "Database Systems",
]

HOGWARTS_STUDY_SUBJECTS = [
    "Transfiguration",
    "Charms",
    "Potions",
    "Defense against the Dark Arts",
    "Herbology",
    "Study of Ancient Runes",
    "Divination",
    "Muggle Studies",
    "Apparition",
    "Arithmancy",
    "Care of Magical Creatures",
]

# Gale–Shapley bipartite split for Hogwarts teammate CSV (domain = frontend/backend on CS datasets).
WORK_PREF_RESEARCHER = "Researcher/ Theoretical work"
WORK_PREF_EXPLORER = "Explorer/ Practical Implementation"


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "regions"
OUTPUT_DIR = BASE_DIR / "outputs"
REGISTRY_PATH = BASE_DIR / "regions.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def region_slug(region: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in region).strip("_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw


def region_csv_path(region: str) -> Path:
    return DATA_DIR / f"{region_slug(region)}.csv"


def resolve_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    if canonical_name in df.columns:
        return canonical_name
    for alias in CANONICAL_ALIASES.get(canonical_name, []):
        if alias in df.columns:
            return alias
    return None


def load_or_materialize_region_df(region: str, region_info: Dict[str, str], cfg: MatchingConfig) -> pd.DataFrame:
    cache_path = region_csv_path(region)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    if region_info.get("type") == "csv":
        src = BASE_DIR / region_info["path"]
        if src.exists():
            df = canonicalize_dataset(pd.read_csv(src), region=region)
        else:
            df = pd.DataFrame(columns=["user_id", "name", "region"])
            src.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(src, index=False)
    else:
        if region == "synthetic_region":
            from matcher import build_synthetic_profiles

            df = build_synthetic_profiles(cfg.n_users, mode="roommate", region=region)
        else:
            df = pd.DataFrame(columns=["user_id", "name", "region"])
    if "region" not in df.columns:
        df["region"] = region
    df = ensure_mode_dataset_columns(df, region, region_info)
    if region_info.get("type") == "csv":
        src = BASE_DIR / region_info["path"]
        src.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(src, index=False)
    df.to_csv(cache_path, index=False)
    return df


def save_region_df(region: str, df: pd.DataFrame, source_csv: Path | None = None) -> None:
    region_csv_path(region).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(region_csv_path(region), index=False)
    if source_csv is not None:
        source_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(source_csv, index=False)


def ensure_mode_dataset_columns(df: pd.DataFrame, region: str, region_info: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    kind = region_info.get("kind", "roommate")
    if kind == "roommate":
        if "duty_preference" not in out.columns:
            # 47% cooking, 53% cleaning as requested.
            rng = random.Random(42)
            n = len(out)
            cutoff = int(round(0.47 * n))
            vals = ["cooking"] * cutoff + ["cleaning"] * max(0, n - cutoff)
            rng.shuffle(vals)
            out["duty_preference"] = vals
    else:
        defaults = {
            "purpose": "passable grades- low time investment",
            "meeting_preference": "both ok",
            "preferred_language": "python",
            "study_interests": "AI-ML",
            "domain": "frontend",
            "work_preference": WORK_PREF_EXPLORER,
        }
        for c in ["purpose", "meeting_preference", "study_interests"]:
            if c not in out.columns:
                out[c] = defaults[c]
        # CS-style teammate CSV uses domain; Hogwarts uses work_preference (mutually exclusive).
        if "domain" not in out.columns and "work_preference" not in out.columns:
            out["domain"] = defaults["domain"]
        if "preferred_language" not in out.columns:
            # Optional: Hogwarts datasets omit this column intentionally.
            pass
        for oc in OCEAN_COLS:
            if oc not in out.columns:
                out[oc] = 5
    return out


# Self-service filter UI: explicit "no restriction" choice (roommate filters default to this).
FILTER_ANY_LABEL = "Any (no filter)"


def columns_for_self_service_filters(df: pd.DataFrame, mode: str) -> List[str]:
    """Roommate self-service: omit noisy / duplicate / identity filters."""
    if mode != "roommate":
        return list(df.columns)
    skip: set[str] = set()
    nc = resolve_column(df, "name")
    uc = resolve_column(df, "user_id")
    if nc:
        skip.add(nc)
    if uc:
        skip.add(uc)
    skip.update(OCEAN_COLS)
    out: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        out.append(col)
    return out


def unique_values(df: pd.DataFrame, col: str) -> List[str]:
    c = resolve_column(df, col)
    if not c:
        return []
    vals = [str(x).strip() for x in df[c].dropna().astype(str).tolist()]
    vals = [v for v in vals if v and v.lower() not in {"nan", "none"}]
    return sorted(set(vals))


def apply_filters_multi(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for col, vals in filters.items():
        if not vals:
            continue
        c = resolve_column(out, col)
        if c:
            out = out[out[c].astype(str).isin([str(v) for v in vals])]
    return out.reset_index(drop=True)


def _is_blank_display_cell(x: object) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return True
    s = str(x).strip()
    return not s or s.lower() in {"nan", "none", "—", "-"}


def _person_display_label(row: pd.Series, name_col: str | None, uid_col: str) -> str:
    if name_col and name_col in row.index and not _is_blank_display_cell(row[name_col]):
        return str(row[name_col]).strip()
    if uid_col in row.index and not _is_blank_display_cell(row[uid_col]):
        return str(row[uid_col]).strip()
    return ""


def user_display_labels(df: pd.DataFrame) -> Dict[str, str]:
    """Map internal user_id string → human-readable label for captions (Name column when present)."""
    uid_col = resolve_column(df, "user_id") or "user_id"
    name_col = resolve_column(df, "name")
    labels: Dict[str, str] = {}
    for _, row in df.iterrows():
        uid = str(row[uid_col]).strip()
        lbl = _person_display_label(row, name_col, uid_col)
        labels[uid] = lbl if lbl else uid
    return labels


def collect_dynamic_self_service(region_df: pd.DataFrame, region_key: str) -> Tuple[Dict[str, object], str]:
    """
    Pick your row from a name-only dropdown (Streamlit selectbox supports type-to-search).
    Every column becomes a selectbox (options = unique values in dataset).
    OCEAN traits: select 1–10. Study interests: CS catalog vs Hogwarts subjects based on dataset name.
    """
    st.subheader("Identify yourself (self-service)")
    uid_col = resolve_column(region_df, "user_id") or "user_id"
    if uid_col not in region_df.columns:
        st.error("This dataset has no user identifier column (`user_id` / `Name`).")
        return {}, ""
    name_col = resolve_column(region_df, "name")

    st.caption("Select your name from the list below (you can type in the box to jump/search).")

    _sk = region_df.apply(lambda r: _person_display_label(r, name_col, uid_col).lower(), axis=1)
    cand = region_df.assign(_sk=_sk).sort_values("_sk").drop(columns=["_sk"])

    base_labels: List[str] = []
    for _, row in cand.iterrows():
        lbl = _person_display_label(row, name_col, uid_col)
        if not lbl:
            lbl = str(row[uid_col]).strip()
        base_labels.append(lbl)
    counts = Counter(base_labels)
    display_options: List[str] = []
    option_uids: List[str] = []
    for (_, row), lbl in zip(cand.iterrows(), base_labels):
        uid = str(row[uid_col]).strip()
        disp = lbl if counts[lbl] <= 1 else f"{lbl} · {uid}"
        display_options.append(disp)
        option_uids.append(uid)

    choice = st.selectbox("Select your name", display_options, key="picked_record")
    uid_chosen = option_uids[display_options.index(choice)]
    base = region_df[region_df[uid_col].astype(str) == uid_chosen].iloc[0]

    st.markdown("**Your profile** — defaults load from your row; change any field via dropdowns.")
    profile: Dict[str, object] = {}

    for col in sorted(region_df.columns):
        raw_val = base[col]
        val_str = "" if pd.isna(raw_val) else str(raw_val).strip()

        if col == uid_col:
            profile[col] = val_str
            st.text_input("Name", value=val_str, disabled=True, key=f"ro_name_{uid_chosen}")
            continue

        if col in OCEAN_COLS:
            ocean_opts = [str(i) for i in range(1, 11)]
            try:
                v = int(float(val_str)) if val_str else 5
            except ValueError:
                v = 5
            v = max(1, min(10, v))
            val_str = str(v)
            ix = ocean_opts.index(val_str) if val_str in ocean_opts else 4
            picked = st.selectbox(col, ocean_opts, index=ix, key=f"p_{uid_chosen}_{col}")
            profile[col] = int(picked)
            continue

        if col == "study_interests":
            if "hogwarts" in region_key.lower():
                opts = HOGWARTS_STUDY_SUBJECTS
            else:
                opts = STUDY_INTERESTS_13
            ix = opts.index(val_str) if val_str in opts else 0
            profile[col] = st.selectbox(col, opts, index=ix, key=f"p_{uid_chosen}_{col}")
            continue

        opts = unique_values(region_df, col)
        if not opts:
            opts = [val_str] if val_str else ["—"]
        ix = opts.index(val_str) if val_str in opts else 0
        profile[col] = st.selectbox(col, opts, index=min(ix, len(opts) - 1), key=f"p_{uid_chosen}_{col}")

    profile.setdefault("region", base.get("region", region_key))
    return profile, uid_chosen


def score_map_from_scores(scores: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    sm = {(str(r["u"]), str(r["v"])): float(r["combined_weight"]) for _, r in scores.iterrows()}
    sm.update({(v, u): w for (u, v), w in list(sm.items())})
    return sm


def gale_pairs_by_sets(df: pd.DataFrame, scores: pd.DataFrame, set_col: str, set_a_val: str, set_b_val: str) -> List[Tuple[str, str, float]]:
    c = resolve_column(df, set_col)
    if not c:
        return []
    set_a = df[df[c].astype(str).str.lower() == set_a_val.lower()]["user_id"].astype(str).tolist()
    set_b = df[df[c].astype(str).str.lower() == set_b_val.lower()]["user_id"].astype(str).tolist()
    if not set_a or not set_b:
        return []
    sm = score_map_from_scores(scores)
    pref_a = {a: sorted(set_b, key=lambda b: sm.get((a, b), 0.0), reverse=True) for a in set_a}
    pref_b = {b: sorted(set_a, key=lambda a: sm.get((a, b), 0.0), reverse=True) for b in set_b}
    pairs = gale_shapley_bipartite(pref_a, pref_b)
    return sorted([(a, b, sm.get((a, b), 0.0)) for a, b in pairs.items()], key=lambda x: x[2], reverse=True)


def _fmt_fb(uid: str, id_labels: Dict[str, str] | None) -> str:
    if id_labels and uid in id_labels and id_labels[uid]:
        return id_labels[uid]
    return uid


def ensure_everyone_paired(
    users: List[str],
    primary_pairs: List[Tuple[str, str, float]],
    scores: pd.DataFrame,
    id_labels: Dict[str, str] | None = None,
) -> Tuple[List[Tuple[str, str, float]], str]:
    used = set()
    out: List[Tuple[str, str, float]] = []
    for a, b, s in primary_pairs:
        used.update([a, b])
        out.append((a, b, s))
    remaining = [u for u in users if u not in used]
    sm = score_map_from_scores(scores)
    fallback_edges: List[Tuple[str, str, float]] = []
    while len(remaining) >= 2:
        a = remaining.pop(0)
        best_idx = max(range(len(remaining)), key=lambda i: sm.get((a, remaining[i]), 0.0))
        b = remaining.pop(best_idx)
        s_ab = sm.get((a, b), 0.0)
        out.append((a, b, s_ab))
        fallback_edges.append((a, b, s_ab))
    if len(remaining) == 1:
        lone = remaining[0]
        others = [u for u in users if u != lone]
        if others:
            best_partner = max(others, key=lambda x: sm.get((lone, x), 0.0))
            s_lb = sm.get((lone, best_partner), 0.0)
            out.append((lone, best_partner, s_lb))
            fallback_edges.append((lone, best_partner, s_lb))
        else:
            out.append((lone, lone, 1.0))
            fallback_edges.append((lone, lone, 1.0))
    if not fallback_edges:
        note = "No fallback needed; algorithm output already covered all users."
    else:
        affected = sorted({u for a, b, _ in fallback_edges for u in (a, b)})
        names_joined = ", ".join(_fmt_fb(u, id_labels) for u in affected)
        pair_bits = [
            f"{_fmt_fb(a, id_labels)} ↔ {_fmt_fb(b, id_labels)} (score {s:.4f})" for a, b, s in fallback_edges
        ]
        note = (
            f"Greedy fallback formed {len(fallback_edges)} pair(s) not produced by the core algorithm alone. "
            f"People involved: {names_joined}. "
            f"Fallback pairs: {'; '.join(pair_bits)}."
        )
    return out, note


def run_irving_with_fallback(
    users: List[str],
    prefs: Dict[str, List[str]],
    scores: pd.DataFrame,
    max_irving_users: int = 140,
    id_labels: Dict[str, str] | None = None,
) -> Tuple[List[Tuple[str, str, float]], str]:
    """
    Returns (final_pairs_covering_all_users, note).
    If Irving fails or is skipped, falls back to greedy completion.
    """
    if len(users) > max_irving_users:
        fallback, fb_note = ensure_everyone_paired(users, [], scores, id_labels)
        return fallback, f"Irving skipped on large pool ({len(users)} users > {max_irving_users}). {fb_note}"

    irving_map = stable_roommates_irving(prefs)
    if not irving_map:
        fallback, fb_note = ensure_everyone_paired(users, [], scores, id_labels)
        return fallback, f"No stable roommate matching from Irving (algorithm returned no solution). {fb_note}"

    irving_pairs = map_pairs_to_scored_list(irving_map, scores)
    completed, completion_note = ensure_everyone_paired(users, irving_pairs, scores, id_labels)
    return completed, f"Irving stable matching found. {completion_note}"


def find_user_match(pairings: List[Tuple[str, str, float]], user_id: str) -> Tuple[str, str, float] | None:
    for a, b, s in pairings:
        if a == user_id or b == user_id:
            return (a, b, s)
    return None


def _execute_matching_pipeline(
    *,
    region_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    region: str,
    registry: Dict[str, Dict[str, str]],
    cfg: MatchingConfig,
    mode: str,
    actor_mode: str,
    new_profile: Dict[str, object],
    focus_user: str | None,
    render_graph: bool,
    graph_max_nodes: int,
    graph_show_only_matches: bool,
    graph_min_edge_weight: float,
    graph_max_edges_per_node: int,
    graph_show_labels: bool,
) -> None:
    uid_col = resolve_column(region_df, "user_id") or "user_id"
    working_df = filtered_df.copy()
    if actor_mode == "self-service":
        if not new_profile or not focus_user:
            st.error("Select your record from this dataset before running.")
            st.stop()
        in_filtered = focus_user in filtered_df[uid_col].astype(str).values
        if not in_filtered:
            st.warning(
                "Your selected profile is outside current filters; it will still be included for scoring "
                "against the filtered candidate pool."
            )
        merged = region_df.copy()
        midx = merged.index[merged[uid_col].astype(str) == focus_user]
        for k, v in new_profile.items():
            merged.loc[midx, k] = v
        src_path = BASE_DIR / registry[region]["path"] if registry[region].get("type") == "csv" else None
        save_region_df(region, merged, source_csv=src_path)
        widx = working_df.index[working_df[uid_col].astype(str) == focus_user]
        if len(widx) == 0:
            # Keep the filtered pool, but always include the selected self profile for cross-group matching.
            working_df = pd.concat([working_df, pd.DataFrame([new_profile])], ignore_index=True)
        else:
            for k, v in new_profile.items():
                working_df.loc[widx, k] = v

    if len(working_df) < 2:
        st.warning("Not enough users to run matching.")
        st.stop()

    graph, scores = build_weighted_graph(working_df, cfg, mode=mode)
    users = working_df["user_id"].astype(str).tolist()
    prefs = build_preference_lists_from_scores(scores, users)
    id_labels = user_display_labels(working_df)

    blossom_pairs = max_weight_roommate_matching(graph)
    if mode == "roommate":
        gs_pairs = gale_pairs_by_sets(working_df, scores, "duty_preference", "cooking", "cleaning")
    elif resolve_column(working_df, "domain"):
        gs_pairs = gale_pairs_by_sets(working_df, scores, "domain", "frontend", "backend")
    elif resolve_column(working_df, "work_preference"):
        gs_pairs = gale_pairs_by_sets(
            working_df,
            scores,
            "work_preference",
            WORK_PREF_RESEARCHER,
            WORK_PREF_EXPLORER,
        )
    else:
        gs_pairs = []

    irving_pairs, irving_note = run_irving_with_fallback(users, prefs, scores, id_labels=id_labels)

    blossom_pairs, blossom_note = ensure_everyone_paired(users, blossom_pairs, scores, id_labels)
    gs_pairs, gs_note = (
        ensure_everyone_paired(users, gs_pairs, scores, id_labels)
        if gs_pairs
        else ensure_everyone_paired(users, [], scores, id_labels)
    )

    heading = "Admin Batch Output" if actor_mode == "admin batch" else "Self-Service Output"
    st.subheader(heading)
    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates", len(users))
    m2.metric("Blossom pairs", len(blossom_pairs))
    m3.metric("Gale-Shapley pairs", len(gs_pairs))
    t1, t2, t3 = st.tabs(["Blossom all pairings", "Gale-Shapley all pairings", "Irving all pairings"])
    t1.dataframe(pd.DataFrame(blossom_pairs, columns=["user_a", "user_b", "score"]))
    t2.dataframe(pd.DataFrame(gs_pairs, columns=["user_a", "user_b", "score"]))
    t3.dataframe(pd.DataFrame(irving_pairs, columns=["user_a", "user_b", "score"]))
    st.caption(f"Blossom completion: {blossom_note}")
    st.caption(f"Gale-Shapley completion: {gs_note}")
    st.caption(f"Irving status: {irving_note}")

    if actor_mode == "self-service" and focus_user:
        st.subheader("Your Match Summary")
        primary_pairs = gs_pairs if gs_pairs else blossom_pairs
        match_row = find_user_match(primary_pairs, focus_user)
        if not match_row:
            st.warning("No pair found for your user in the selected algorithm output.")
        else:
            a, b, s = match_row
            partner = b if a == focus_user else a
            c = working_df.copy()
            user_meta = c[c["user_id"].astype(str) == focus_user].head(1)
            partner_meta = c[c["user_id"].astype(str) == partner].head(1)
            st.write(f"Matched user: `{partner}` with compatibility score `{s:.4f}`")
            meta_cols = [
                col
                for col in [
                    "name",
                    "gender",
                    "diet",
                    "sleep_type",
                    "cleanliness",
                    "noise_preference",
                    "domain",
                    "work_preference",
                    "preferred_language",
                    "meeting_preference",
                ]
                if col in c.columns
            ]
            if not user_meta.empty:
                st.write("Your profile snapshot:")
                st.dataframe(user_meta[["user_id"] + meta_cols])
            if not partner_meta.empty:
                st.write("Matched profile snapshot:")
                st.dataframe(partner_meta[["user_id"] + meta_cols])

    if render_graph:
        highlight = blossom_pairs if mode == "roommate" else (gs_pairs if gs_pairs else blossom_pairs)
        graph_file = OUTPUT_DIR / f"{mode}_{actor_mode.replace(' ', '_')}_graph_ui.html"
        export_interactive_graph(
            graph,
            graph_file,
            highlight_pairs=highlight,
            focus_user=focus_user,
            max_nodes=graph_max_nodes,
            show_only_highlight_edges=graph_show_only_matches,
            min_edge_weight=graph_min_edge_weight,
            max_edges_per_node=graph_max_edges_per_node,
            show_labels=graph_show_labels,
        )
        st.subheader("Interactive Graph")
        try:
            html_body = graph_file.read_text(encoding="utf-8")
            components.html(html_body, height=820, scrolling=True)
        except OSError as exc:
            st.error(f"Could not read graph file: {graph_file} ({exc})")
        st.caption(f"Saved copy: `{graph_file}` (open in browser if the embedded view fails).")
    else:
        st.caption("Graph rendering is off for speed. Enable it from sidebar when needed.")

    with st.expander("How algorithms are used here"):
        st.markdown(
            "- **Gale-Shapley**: preference lists are derived by sorting compatibility scores for each user; then run bipartite matching.\n"
            "  - Roommate bipartite sets: `duty_preference = cooking` vs `cleaning`\n"
            "  - Teammate bipartite sets: `domain = frontend` vs `backend` (CS class), or "
            "`work_preference` researcher vs explorer (Hogwarts).\n"
            "- **Irving stable roommates** (roommate mode): one-sided preference lists from score ranking across all users; returns stable matching if one exists."
        )

    with st.expander("How greedy fallback completion works (same helper for all three)", expanded=False):
        st.markdown(
            """
After **Blossom**, **Gale–Shapley**, or **Irving** runs, the app calls one shared routine (`ensure_everyone_paired`) so **every user appears in some pair** in the tables you see.

**What the algorithms leave behind**

- **Blossom:** Only pairs people who have an edge *above your similarity threshold*. Anyone with no qualifying edge is unmatched until fallback.
- **Gale–Shapley:** Matches within two sides (e.g. cooking vs cleaning). People on one side can outnumber the other, or a side can be empty—then GS does not cover everyone.
- **Irving:** Either returns a stable matching or **nothing**. If it fails (or the pool is larger than the Irving limit), the app discards Irving’s structure and builds pairs using fallback only.

**What fallback does (simple)**

1. Take whoever is **already paired** by that algorithm and keep those pairs.
2. Among everyone **still unmatched**, repeatedly take the next person `a`, find another unmatched person `b` with the **highest compatibility score** `combined_weight` with `a`, and pair them.
3. If one person is left over (odd pool), pair them with whoever in the **whole cohort** has the best score with them (same helper as today).

So fallback is **not** another Blossom/GS/Irving pass—it is **greedy completion using your precomputed similarity scores**, so you always get a full pairing list for display. Names shown in the completion captions are resolved from the dataset’s name column when available.
"""
        )

    with st.expander("How Blossom (max-weight matching) works", expanded=False):
        st.markdown(
            """
Blossom here is **not** a greedy procedure that repeatedly picks the highest-weight edge from a shrinking pool.

NetworkX’s **max-weight matching** on the similarity graph finds a **matching** (each person paired with at most one partner) that **maximizes the sum of edge weights** of all chosen pairs **at once**—subject to the constraint that no person appears in more than one pair. That is a global optimization (implemented with Edmonds’ blossom algorithm), not “best edge first, then best among what’s left,” so it does **not** inherently mean that later pairs are worse because earlier steps “used up” the good similarity: edges are only comparable through the **joint** choice of a full matching.

In practice, if your **edge threshold** hides many medium edges, the graph can become sparse and the optimum sum may still pair some people with moderate scores—that reflects missing edges, not Blossom taking low edges because it ran out of high ones in a sequential sense.

**Compared to your fallback step:** after Blossom, the app may **complete** unmatched users by greedy best remaining score—that *is* sequential and can use weaker edges for stragglers. That is separate from Blossom’s global optimum on the matched subgraph.
"""
        )

    results_payload = {
        "actor_mode": actor_mode,
        "mode": mode,
        "region": region,
        "blossom_pairs": [{"user_a": a, "user_b": b, "score": s} for a, b, s in blossom_pairs],
        "gale_shapley_pairs": [{"user_a": a, "user_b": b, "score": s} for a, b, s in gs_pairs],
        "irving_pairs": [{"user_a": a, "user_b": b, "score": s} for a, b, s in irving_pairs],
        "blossom_note": blossom_note,
        "gale_shapley_note": gs_note,
        "irving_note": irving_note,
    }
    with open(OUTPUT_DIR / f"{mode}_{actor_mode.replace(' ', '_')}_results_ui.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    st.success("Matching finished.")


def render_dataset_schema(mode: str, registry: Dict[str, Dict[str, str]]) -> None:
    """Professor-friendly summary: files, registry, filter vs score columns, algorithms."""
    roommate_ds = sorted([k for k, v in registry.items() if v.get("kind", "roommate") == "roommate"])
    teammate_ds = sorted([k for k, v in registry.items() if v.get("kind", "roommate") == "teammate"])
    with st.expander("Dataset schema & pipeline (for grading)", expanded=False):
        st.markdown(
            f"""
**Where data lives**
- **Registry:** `{REGISTRY_PATH}` — each key is a selectable dataset; fields include `type` (`csv` / `synthetic`), `path` (source CSV relative to project root), `kind` (`roommate` vs `teammate`).
- **Cached copy per selection:** `{DATA_DIR}/<region_key>.csv` — materialized after first load (canonical columns + any auto-filled columns).

**Roommate datasets** (`kind=roommate`): {", ".join(roommate_ds) or "(none)"}
- Example source: `Girls_pg_hostel_CSV_data-1.csv` — person id may be a column named `Name` or `user_id` (canonicalized to `user_id`); roommate-relevant columns as in the CSV.
- **Bipartite Gale–Shapley sets:** `duty_preference` → `cooking` vs `cleaning` (47% / 53% filled on hostel data if that column was missing).

**Teammate datasets** (`kind=teammate`): {", ".join(teammate_ds) or "(none)"}
- **Computer Science class CSV:** `user_id`, `name`, `purpose`, `meeting_preference`, `preferred_language`, `study_interests` (13 catalog values in UI), `domain` (`frontend`/`backend`), `region`, OCEAN traits.
- **Hogwarts CSV:** same purpose/meeting options as CS except no hackathon purpose; **no** `preferred_language`; wizarding `study_interests`; `work_preference` (`Researcher/ Theoretical work` vs `Explorer/ Practical Implementation`) instead of `domain`; OCEAN traits.
- **Bipartite Gale–Shapley:** CS uses `domain` → frontend vs backend; Hogwarts uses `work_preference` → researcher vs explorer.

**Admin vs self-service**
- **Admin batch:** no preference filters — all rows in the selected dataset are candidates.
- **Self-service:** **Hard filters** (multi-select per column; OR within a column, AND across columns) narrow rows first (they do **not** enter the similarity vector for roommate mode; teammate filters overlap with columns that *also* contribute to scoring — see below). You pick your row from the name list; edits update that row in the CSV cache and source file; matching runs on the filtered pool.

**How compatibility scores are built** (`matcher.build_weighted_graph`)
1. **Numeric columns** (e.g. OCEAN 1–10, `social_energy_rating`): scaled with `MinMaxScaler`; **cosine** and **Euclidean**-style similarity on that block.
2. **Categorical columns** (non-multi text): one-hot encoded; included in **cosine** block.
3. **Multi-value text** (comma-separated): token sets → **Jaccard** between pairs.
4. **Combined edge weight:** `w_cos * cosine + w_jac * jaccard + w_euc * euclidean_sim` (sidebar sliders, renormalized to sum to 1).
5. **Graph:** edge if combined ≥ **edge threshold**; Blossom uses this graph’s `weight`.

**Algorithms (same weights, different objectives)**
- **Blossom (max-weight matching):** maximize sum of selected edge weights over a matching (NetworkX).
- **Gale–Shapley:** for each side of the bipartite split, preference order = other side sorted by pairwise `combined_weight`; then classic propose/reject.
- **Irving (stable roommates):** each person’s list = all others sorted by `combined_weight`; Irving phases; if no stable outcome or pool too large, **fallback** completes everyone using best remaining scores (no self-pair unless literally one user in the world set).

**Roommate scoring columns** (after filters): social energy, interests, room type, privacy, duty preference, OCEAN — *not* gender/diet/sleep/clean/noise/smoking as score inputs (those are filter-only in UI).
**Teammate scoring columns** (present columns only): purpose, meeting preference, study interests, OCEAN; CS adds `domain` and `preferred_language`; Hogwarts adds `work_preference` instead of those two; optional extras if present (working style, goal topic, etc.).
"""
        )


def main() -> None:
    st.set_page_config(page_title="Graph Matcher UI", layout="wide")
    st.title("Roommate / Teammate Graph Matcher")
    cfg = MatchingConfig(output_dir="outputs")
    registry = ensure_registry(REGISTRY_PATH)

    with st.sidebar:
        st.header("Run Setup")
        actor_mode = st.selectbox("Usage mode", ["admin batch", "self-service"])
        mode_label = st.selectbox("What do you want to do?", ["find roommate", "find teammate"])
        mode = "roommate" if mode_label == "find roommate" else "teammate"
        if mode == "roommate":
            region_names = sorted([k for k, v in registry.items() if v.get("kind", "roommate") == "roommate"])
        else:
            region_names = sorted([k for k, v in registry.items() if v.get("kind", "roommate") == "teammate"])
        if not region_names:
            st.error("No datasets available for selected mode.")
            st.stop()
        region = st.selectbox("Region dataset", region_names)
        render_graph = st.checkbox("Render interactive graph (slower on large data)", value=False)
        graph_max_nodes = st.slider("Graph node cap (if rendering)", 50, 400, 150, 10)
        graph_show_only_matches = False
        graph_min_edge_weight = 0.0
        graph_max_edges_per_node = 5
        graph_show_labels = True
        if render_graph:
            st.markdown("**Graph UX controls**")
            detail_mode = st.selectbox("Graph detail level", ["Simple (fast)", "Balanced", "Detailed"], index=1)
            default_edge_cap = 2 if detail_mode == "Simple (fast)" else (5 if detail_mode == "Balanced" else 10)
            default_labels = detail_mode != "Simple (fast)"
            default_matches_only = detail_mode == "Simple (fast)"
            graph_show_only_matches = st.checkbox("Show only final matched pairs", value=default_matches_only)
            graph_show_labels = st.checkbox("Show node labels", value=default_labels)
            graph_max_edges_per_node = st.slider("Max edges per node in graph", 1, 20, default_edge_cap, 1)
            graph_min_edge_weight = st.slider("Visualization min edge weight", 0.0, 1.0, 0.0, 0.01)
        cfg.edge_threshold = st.slider("Edge threshold", 0.0, 1.0, float(cfg.edge_threshold), 0.01)
        w1 = st.slider("Weight: Cosine", 0.0, 1.0, float(cfg.weight_cosine), 0.05)
        w2 = st.slider("Weight: Jaccard", 0.0, 1.0, float(cfg.weight_jaccard), 0.05)
        w3 = st.slider("Weight: Euclidean", 0.0, 1.0, float(cfg.weight_euclidean), 0.05)
        denom = w1 + w2 + w3 if (w1 + w2 + w3) > 0 else 1.0
        cfg.weight_cosine, cfg.weight_jaccard, cfg.weight_euclidean = w1 / denom, w2 / denom, w3 / denom

    region_df = canonicalize_dataset(load_or_materialize_region_df(region, registry[region], cfg), region=region)
    st.write(f"Loaded region `{region}` with `{len(region_df)}` profiles.")
    render_dataset_schema(mode, registry)

    if actor_mode == "admin batch":
        st.info("Admin mode uses the full selected region dataset (no candidate preference filters).")
        filtered_df = region_df.copy()
    else:
        with st.expander("Preferences / Filters (multi-select, self-service only)", expanded=True):
            if mode == "roommate":
                st.markdown(
                    "**Hard filters:** AND across columns; OR within each column. "
                    "Each field defaults to **Any (no filter)**. "
                    "Roommate mode hides name, user_id, and OCEAN traits (duty uses the single `duty_preference` column)."
                )
            else:
                st.markdown(
                    "**Hard filters:** only rows that satisfy **all** active filters (AND). "
                    "Each field defaults to **Any (no filter)**."
                )
            filter_state: Dict[str, List[str]] = {}
            for col in columns_for_self_service_filters(region_df, mode):
                opts = unique_values(region_df, col)
                if not opts:
                    continue
                ui_opts = [FILTER_ANY_LABEL] + opts
                chosen = st.multiselect(
                    f"Filter: {col}",
                    options=ui_opts,
                    default=[FILTER_ANY_LABEL],
                    key=f"mflt_{col}",
                )
                effective = [v for v in chosen if v != FILTER_ANY_LABEL]
                if effective:
                    filter_state[col] = effective
            filtered_df = apply_filters_multi(region_df, filter_state)
            st.caption(f"Profiles after filter: {len(filtered_df)}")

    focus_user: str | None = None
    new_profile: Dict[str, object] = {}
    if actor_mode == "self-service":
        new_profile, focus_user = collect_dynamic_self_service(region_df, region)

    if not st.button("Run Matching", type="primary"):
        st.stop()

    with st.spinner(
        "Running matching (Blossom, Gale–Shapley, Irving). Please wait — results appear when this finishes. "
        "Avoid clicking Run Matching again while this runs."
    ):
        _execute_matching_pipeline(
            region_df=region_df,
            filtered_df=filtered_df,
            region=region,
            registry=registry,
            cfg=cfg,
            mode=mode,
            actor_mode=actor_mode,
            new_profile=new_profile,
            focus_user=focus_user,
            render_graph=render_graph,
            graph_max_nodes=graph_max_nodes,
            graph_show_only_matches=graph_show_only_matches,
            graph_min_edge_weight=graph_min_edge_weight,
            graph_max_edges_per_node=graph_max_edges_per_node,
            graph_show_labels=graph_show_labels,
        )


if __name__ == "__main__":
    main()
