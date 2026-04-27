from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

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


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "regions"
OUTPUT_DIR = BASE_DIR / "outputs"
REGISTRY_PATH = BASE_DIR / "regions.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def region_csv_path(region: str) -> Path:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in region).strip("_")
    return DATA_DIR / f"{safe}.csv"


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
    df.to_csv(cache_path, index=False)
    return df


def save_region_df(region: str, df: pd.DataFrame) -> None:
    region_csv_path(region).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(region_csv_path(region), index=False)


def unique_values(df: pd.DataFrame, col: str) -> List[str]:
    c = resolve_column(df, col)
    if not c:
        return []
    vals = [str(x).strip() for x in df[c].dropna().astype(str).tolist()]
    vals = [v for v in vals if v and v.lower() not in {"nan", "none"}]
    return sorted(set(vals))


def apply_filters(df: pd.DataFrame, filters: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, val in filters.items():
        c = resolve_column(out, col)
        if c and val and val != "Any":
            out = out[out[c].astype(str).str.lower() == val.lower()]
    return out.reset_index(drop=True)


def collect_profile_form(mode: str, selected_region: str, pref_defaults: Dict[str, str]) -> Dict[str, object]:
    st.subheader("Create Your Profile (required in self-service)")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Name", value="new_user")
        user_id = st.text_input("User ID (optional)")
        st.text_input("Region", value=selected_region, disabled=True)
        gender = st.selectbox("Gender", ["female", "male", "other", "prefer_not_to_say"])
        diet = st.selectbox("Diet", ["vegetarian", "non-vegetarian", "vegan", "jain", "no restrictions"])
        sleep_type = st.selectbox("Sleep type", ["light sleeper", "heavy sleeper", "mixed"])
        cleanliness = st.selectbox("Cleanliness", ["messy", "both", "organized"])
    with c2:
        noise_preference = st.selectbox("Noise preference", ["quiet", "normal", "noisy"])
        smoking_drinking = st.selectbox(
            "Smoking/Drinking",
            ["non-smoker/non-drinker", "smoker", "drinker", "both", "okay with roommate habits"],
        )
        social_energy_rating = st.slider("Social energy", 1, 10, 5)
        interests = st.text_input("Interests (comma-separated)", value=pref_defaults.get("study_interests", "music,coding"))
        openness = st.slider("Openness", 1, 10, 5)
        conscientiousness = st.slider("Conscientiousness", 1, 10, 5)
        extraversion = st.slider("Extraversion", 1, 10, 5)
        agreeableness = st.slider("Agreeableness", 1, 10, 5)
        neuroticism = st.slider("Neuroticism", 1, 10, 5)
    profile: Dict[str, object] = {
        "name": name,
        "user_id": user_id.strip() if user_id.strip() else "",
        "region": selected_region,
        "gender": gender,
        "diet": diet,
        "sleep_type": sleep_type,
        "cleanliness": cleanliness,
        "noise_preference": noise_preference,
        "smoking_drinking": smoking_drinking,
        "social_energy_rating": social_energy_rating,
        "interests": interests,
        "openness": openness,
        "conscientiousness": conscientiousness,
        "extraversion": extraversion,
        "agreeableness": agreeableness,
        "neuroticism": neuroticism,
    }
    if mode == "roommate":
        profile["duty_preference"] = st.selectbox("Duty preference (bipartite set)", ["cooking", "cleaning"])
        profile["room_type_preference"] = st.selectbox("Room type", ["private", "shared"])
    else:
        profile["goal_topic"] = st.text_input("Teammate purpose/topic", value="daa project")
        profile["meeting_preference"] = st.selectbox("Meeting preference", ["online", "offline", "hybrid"])
        profile["domain"] = st.selectbox("Domain (bipartite set)", ["frontend", "backend"])
        profile["preferred_language"] = st.selectbox("Preferred language", ["python", "java", "cpp", "javascript"])
        profile["study_interests"] = st.text_input("Study interests", value=pref_defaults.get("study_interests", "ai-ml,algorithms"))
    return profile


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


def ensure_everyone_paired(users: List[str], primary_pairs: List[Tuple[str, str, float]], scores: pd.DataFrame) -> Tuple[List[Tuple[str, str, float]], str]:
    used = set()
    out = []
    for a, b, s in primary_pairs:
        used.update([a, b])
        out.append((a, b, s))
    remaining = [u for u in users if u not in used]
    sm = score_map_from_scores(scores)
    fallback_count = 0
    while len(remaining) >= 2:
        a = remaining.pop(0)
        best_idx = max(range(len(remaining)), key=lambda i: sm.get((a, remaining[i]), 0.0))
        b = remaining.pop(best_idx)
        out.append((a, b, sm.get((a, b), 0.0)))
        fallback_count += 1
    if len(remaining) == 1:
        # No self-pair unless there are truly no alternatives.
        lone = remaining[0]
        others = [u for u in users if u != lone]
        if others:
            best_partner = max(others, key=lambda x: sm.get((lone, x), 0.0))
            out.append((lone, best_partner, sm.get((lone, best_partner), 0.0)))
            fallback_count += 1
        else:
            out.append((lone, lone, 1.0))
            fallback_count += 1
    note = (
        f"Fallback used for {fallback_count} assignment(s): "
        "when strict algorithm pairs were incomplete, best-score completion was applied."
        if fallback_count > 0
        else "No fallback needed; algorithm output already covered all users."
    )
    return out, note


def run_irving_with_fallback(
    users: List[str],
    prefs: Dict[str, List[str]],
    scores: pd.DataFrame,
    max_irving_users: int = 140,
) -> Tuple[List[Tuple[str, str, float]], str]:
    """
    Returns (final_pairs_covering_all_users, note).
    If Irving fails or is skipped, falls back to greedy completion.
    """
    if len(users) > max_irving_users:
        fallback, _ = ensure_everyone_paired(users, [], scores)
        return fallback, "Irving skipped on large pool; used complete fallback pairing."

    irving_map = stable_roommates_irving(prefs)
    if not irving_map:
        fallback, _ = ensure_everyone_paired(users, [], scores)
        return fallback, "No stable matching from Irving; used complete fallback pairing."

    irving_pairs = map_pairs_to_scored_list(irving_map, scores)
    completed, completion_note = ensure_everyone_paired(users, irving_pairs, scores)
    return completed, f"Irving stable matching found. {completion_note}"


def find_user_match(pairings: List[Tuple[str, str, float]], user_id: str) -> Tuple[str, str, float] | None:
    for a, b, s in pairings:
        if a == user_id or b == user_id:
            return (a, b, s)
    return None


def main() -> None:
    st.set_page_config(page_title="Graph Matcher UI", layout="wide")
    st.title("Roommate / Teammate Graph Matcher")
    cfg = MatchingConfig(output_dir="outputs")
    registry = ensure_registry(REGISTRY_PATH)
    region_names = sorted(registry.keys())

    with st.sidebar:
        st.header("Run Setup")
        actor_mode = st.selectbox("Usage mode", ["admin batch", "self-service"])
        mode_label = st.selectbox("What do you want to do?", ["find roommate", "find teammate"])
        mode = "roommate" if mode_label == "find roommate" else "teammate"
        region = st.selectbox("Region dataset", region_names)
        render_graph = st.checkbox("Render interactive graph (slower on large data)", value=False)
        graph_max_nodes = st.slider("Graph node cap (if rendering)", 50, 400, 150, 10)
        cfg.edge_threshold = st.slider("Edge threshold", 0.0, 1.0, float(cfg.edge_threshold), 0.01)
        w1 = st.slider("Weight: Cosine", 0.0, 1.0, float(cfg.weight_cosine), 0.05)
        w2 = st.slider("Weight: Jaccard", 0.0, 1.0, float(cfg.weight_jaccard), 0.05)
        w3 = st.slider("Weight: Euclidean", 0.0, 1.0, float(cfg.weight_euclidean), 0.05)
        denom = w1 + w2 + w3 if (w1 + w2 + w3) > 0 else 1.0
        cfg.weight_cosine, cfg.weight_jaccard, cfg.weight_euclidean = w1 / denom, w2 / denom, w3 / denom

    region_df = canonicalize_dataset(load_or_materialize_region_df(region, registry[region], cfg), region=region)
    st.write(f"Loaded region `{region}` with `{len(region_df)}` profiles.")

    pref_defaults: Dict[str, str] = {}
    if actor_mode == "admin batch":
        st.info("Admin mode uses the full selected region dataset (no candidate preference filters).")
        filtered_df = region_df.copy()
    else:
        with st.expander("Preferences / Filters", expanded=True):
            st.markdown("**Hard Filters (candidate eligibility only):** used only to narrow who is considered.")
            if mode == "roommate":
                filters = {
                    "gender": st.selectbox("Preferred gender", ["Any"] + unique_values(region_df, "gender")),
                    "diet": st.selectbox("Preferred diet", ["Any"] + unique_values(region_df, "diet")),
                    "smoking_drinking": st.selectbox("Smoking/Drinking", ["Any"] + unique_values(region_df, "smoking_drinking")),
                    "sleep_type": st.selectbox("Sleep type", ["Any"] + unique_values(region_df, "sleep_type")),
                    "cleanliness": st.selectbox("Cleanliness", ["Any"] + unique_values(region_df, "cleanliness")),
                    "noise_preference": st.selectbox("Noise preference", ["Any"] + unique_values(region_df, "noise_preference")),
                }
                pref_defaults = {k: v for k, v in filters.items() if v != "Any"}
                filtered_df = apply_filters(region_df, filters)
                st.markdown(
                    "**Scoring qualities (not filters):** social energy, interests, duty preference, room type/privacy, OCEAN traits."
                )
            else:
                filters = {
                    "sleep_type": st.selectbox("Sleep schedule", ["Any"] + unique_values(region_df, "sleep_type")),
                    "meeting_preference": st.selectbox("Meeting preference", ["Any"] + unique_values(region_df, "meeting_preference")),
                    "domain": st.selectbox("Domain", ["Any"] + unique_values(region_df, "domain")),
                    "preferred_language": st.selectbox("Preferred language", ["Any"] + unique_values(region_df, "preferred_language")),
                }
                study_interest = st.text_input("Study interests contain", value="")
                pref_defaults = {"study_interests": study_interest}
                filtered_df = apply_filters(region_df, filters)
                c = resolve_column(filtered_df, "study_interests") or resolve_column(filtered_df, "interests")
                if study_interest.strip() and c:
                    filtered_df = filtered_df[filtered_df[c].astype(str).str.lower().str.contains(study_interest.strip().lower(), na=False)]
                filtered_df = filtered_df.reset_index(drop=True)
                st.markdown(
                    "**Scoring qualities (not filters):** domain, preferred language, study interests, working style/time commitment, goal topic, interests, social energy, OCEAN traits."
                )
            st.caption(f"Profiles after filter: {len(filtered_df)}")

    focus_user = None
    if actor_mode == "self-service":
        new_profile = collect_profile_form(mode, region, pref_defaults)
    else:
        new_profile = None

    if not st.button("Run Matching", type="primary"):
        st.stop()

    working_df = filtered_df.copy()
    if actor_mode == "self-service":
        if not new_profile:
            st.error("Profile is required for self-service.")
            st.stop()
        if not new_profile.get("user_id"):
            new_profile["user_id"] = f"{region[:3].upper()}_SELF_{len(region_df)+1:04d}"
        focus_user = str(new_profile["user_id"])
        merged = pd.concat([region_df, pd.DataFrame([new_profile])], ignore_index=True)
        save_region_df(region, merged)
        working_df = pd.concat([working_df, pd.DataFrame([new_profile])], ignore_index=True)

    if len(working_df) < 2:
        st.warning("Not enough users to run matching.")
        st.stop()

    graph, scores = build_weighted_graph(working_df, cfg, mode=mode)
    users = working_df["user_id"].astype(str).tolist()
    prefs = build_preference_lists_from_scores(scores, users)

    blossom_pairs = max_weight_roommate_matching(graph)
    if mode == "roommate":
        gs_pairs = gale_pairs_by_sets(working_df, scores, "duty_preference", "cooking", "cleaning")
    else:
        gs_pairs = gale_pairs_by_sets(working_df, scores, "domain", "frontend", "backend")

    irving_pairs, irving_note = run_irving_with_fallback(users, prefs, scores)

    # Show full pairings in both modes; self-service additionally highlights created user's match.
    blossom_pairs, blossom_note = ensure_everyone_paired(users, blossom_pairs, scores)
    gs_pairs, gs_note = ensure_everyone_paired(users, gs_pairs, scores) if gs_pairs else ensure_everyone_paired(users, [], scores)

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
            meta_cols = [col for col in ["name", "gender", "diet", "sleep_type", "cleanliness", "noise_preference", "domain", "preferred_language", "meeting_preference"] if col in c.columns]
            if not user_meta.empty:
                st.write("Your profile snapshot:")
                st.dataframe(user_meta[["user_id"] + meta_cols])
            if not partner_meta.empty:
                st.write("Matched profile snapshot:")
                st.dataframe(partner_meta[["user_id"] + meta_cols])

    if render_graph:
        highlight = blossom_pairs if mode == "roommate" else (gs_pairs if gs_pairs else blossom_pairs)
        graph_file = OUTPUT_DIR / f"{mode}_{actor_mode.replace(' ', '_')}_graph_ui.html"
        export_interactive_graph(graph, graph_file, highlight_pairs=highlight, focus_user=focus_user, max_nodes=graph_max_nodes)
        st.subheader("Interactive Graph")
        st.iframe(graph_file.as_uri(), height=820)
    else:
        st.caption("Graph rendering is off for speed. Enable it from sidebar when needed.")

    with st.expander("How algorithms are used here"):
        st.markdown(
            "- **Gale-Shapley**: preference lists are derived by sorting compatibility scores for each user; then run bipartite matching.\n"
            "  - Roommate bipartite sets: `duty_preference = cooking` vs `cleaning`\n"
            "  - Teammate bipartite sets: `domain = frontend` vs `backend`\n"
            "- **Irving stable roommates** (roommate mode): one-sided preference lists from score ranking across all users; returns stable matching if one exists."
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


if __name__ == "__main__":
    main()
