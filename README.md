# Graph-Based Roommate/Dating Matching System  
**Course project (one semester) — Team of 2**

## 1. Problem statement
Finding compatible roommates (or dating partners) is hard because “compatibility” depends on multiple lifestyle and personal factors that aren’t captured well by simple profiles. We model each person as a **vertex** in a graph, and pairwise compatibility as **weighted edges**. We then study and compare algorithms that produce **stable** or **high-compatibility** matchings.

## 2. Key idea
- Build a **compatibility scoring function** from user attributes (e.g., OCEAN traits, sleep schedule, dietary restrictions, cleaning habits, lifestyle choices).
- Convert scores into:
  - **Preference lists** (for stable-matching style algorithms), or
  - **Edge weights** (for maximum-weight matching).
- Compare multiple matching algorithms under consistent datasets and evaluation metrics.

## 3. Algorithms to implement and compare
1. **Gale–Shapley Stable Matching** (stable marriage / admissions style) 
2. **Irving’s Stable Roommates Algorithm** (non-bipartite stable roommates)
3. **Edmonds’ Blossom Algorithm** for **Maximum Weight Matching** 

### Why these three?
- Stable matching focuses on **no blocking pairs** (stability), not necessarily maximum total compatibility. 
- Maximum weight matching maximizes **total compatibility**, but may not be stable. 

## 4. Data & compatibility scoring
### Attributes (initial set)
- **Personality**: OCEAN (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Dietary restrictions
- Sleep schedule 
- Cleaning habits 
- Lifestyle choices 

### Scoring function (proposal-level spec)
Define compatibility between people *i* and *j* as:

\[
s(i,j) = \sum_k w_k \cdot sim_k(i,j)
\]

Where:
- \(sim_k\) is a similarity (or penalty) function for attribute \(k\)
- \(w_k\) are tunable weights (baseline: equal weights; later: sensitivity analysis)

**Deliverable:** clearly document each \(sim_k\) and how it maps raw attributes → [0,1] similarity (or negative penalty).

## 5. Evaluation plan
We will evaluate algorithms on: 
- **Stability:** existence of blocking pairs / stable outcome (where applicable)
- **Total compatibility:** sum of edge weights in the produced matching
- **Runtime / scalability:** time vs. number of users
- **Consistency & reproducibility:** stability of results across repeated runs / tie-breaking

### Suggested experiments (semester-sized)
- Vary number of participants: 50, 100, 250, 500, 1000
- Test multiple distributions for attributes (real dataset + synthetic controlled)
- Sensitivity to scoring weights: adjust \(w_k\) and observe match changes

## 6. Visualization
- Visualize the compatibility network (nodes, weighted edges)
- Visualize final matchings (highlight matched pairs)
- Optional: show heatmaps of compatibility and stability metrics across scenarios 

## 7. Tools / stack
- **Python**
- **NetworkX** (graph modeling)
- **NumPy / Pandas** (data)
- **Matplotlib** (plots/visualization) 

## 8. Dataset plan
- Use public datasets (e.g., Kaggle roommate/dating profile datasets) and
- Generate **synthetic profiles** with controlled attribute distributions for stress-testing. 

## 9. Roles (2-person split)
**Person A (Theory + algorithm correctness)**
- Formalize stability / blocking pairs
- Implement Gale–Shapley + Irving
- Write up algorithm comparison and complexity discussion

**Person B (Scoring + evaluation + visuals)**
- Implement scoring pipeline + preprocessing
- Implement Blossom / max-weight matching
- Run experiments, plots, and visualizations

(We’ll both review each other’s code and co-author the final presentation.)

## 10. Deliverables (by end of semester)
- Working prototype: input profiles → compatibility graph → matching outputs
- Comparative analysis across algorithms (stability vs. compatibility vs. runtime)
- Visualizations of graphs and matchings
- Slide deck + short report (method + results + discussion)

## 11. References
- Gale, D. & Shapley, L. S. (1962). *College admissions and the stability of marriage.*
- Irving, R. W. (1985). *An efficient algorithm for the stable roommates problem.* 
- Edmonds, J. (1965). *Paths, trees, and flowers.* 
- Cormen et al. (2009). *Introduction to Algorithms (3rd ed.).* 
