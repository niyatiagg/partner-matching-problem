# Sprint 1 Progress Report
## DAA Final Project: Graph-Based Roommate/Teammate Matcher

**Team Members:** Niyati Aggarwal, Anaya Dandekar  
**Course:** Design and Analysis of Algorithms  
**Sprint:** Sprint 1 (Foundation + Prototype)  
**Date:** April 2026

### 1) Project Goal
The project models each user as a node in a graph and computes compatibility between users as weighted edges. The primary objective is to generate high-quality matches for roommate selection (and optionally teammate matching) using graph algorithms and similarity metrics.

### 2) What Has Been Implemented
We completed a working Python prototype with an end-to-end pipeline:

1. **Data layer**
   - Synthetic user profile generator for controlled experiments.
   - Optional CSV input support for real datasets or manually collected profiles.
   - Profile schema includes OCEAN traits, lifestyle categories, and interests.

2. **Edge-weight computation**
   - Implemented three similarity components:
     - **Cosine similarity** (normalized numeric + encoded categorical features)
     - **Jaccard similarity** (interests and lifestyle token overlap)
     - **Euclidean-distance-based similarity** on OCEAN traits (`1/(1+d)`)
   - Combined weighted score:
     - `0.45 * cosine + 0.25 * jaccard + 0.30 * euclidean_similarity`
   - Added edge-threshold filtering to remove weak connections and reduce graph noise.

3. **Matching algorithms**
   - **Roommate matching:** Maximum Weight Matching (Blossom through NetworkX).
   - **Teammate mode:** Gale-Shapley Stable Matching over bipartite partition.
   - Outputs include ranked match pairs with compatibility scores.

4. **Visualization and outputs**
   - Interactive graph visualization generated as HTML (PyVis).
   - Exported artifacts:
     - `profiles_used.csv`
     - `pair_scores.csv`
     - `roommate_matching.json`
     - `teammate_matching.json`
     - `compatibility_graph.html`

### 3) Why This Is Relevant to DAA
- The project directly applies graph modeling and weighted matching optimization.
- It compares objective functions and algorithmic behavior (maximum total compatibility vs stability-focused matching).
- It enables complexity/performance analysis with increasing input sizes.

### 4) Preliminary Run Results
A successful test run was executed with 20 synthetic users and threshold 0.45:
- Total users: 20
- Edges retained in compatibility graph: 67
- Roommate pairs (max-weight matching): 10
- Teammate pairs (Gale-Shapley): 10
- Top roommate pair score observed: ~0.627

These outputs confirm that the full pipeline (data -> weighting -> matching -> visualization) is operational.

### 5) Current Status and Risks
**Completed:** Core implementation and baseline validation.  
**Open items:** Dataset quality and evaluation depth still need expansion.  
**Known risk:** Stable roommates (Irving) is not yet implemented; currently represented through Blossom and Gale-Shapley modes.

### 6) Sprint 2 Plan
1. Integrate real dataset(s) and profile cleaning pipeline.
2. Add systematic experiments across user counts (20, 40, 80, 160).
3. Compare weighting variants and thresholds.
4. Add quality metrics (total score, average score, optional stability checks).
5. (Stretch) Implement Irving's Stable Roommates for one-sided stability comparison.
6. Prepare final charts, analysis, and demo narrative.

### 7) Work Division Going Forward
- **Member 1:** Algorithm tuning, complexity analysis, metrics and comparisons.
- **Member 2:** Data preparation, visualization polishing, report writing and presentation.

This sprint establishes a functional, extensible algorithmic base and aligns the implementation with both project goals and instructor feedback on similarity-based edge weighting.
