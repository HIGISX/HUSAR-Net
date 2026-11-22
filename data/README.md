# Datasets

Datasets for *Hierarchical spatial resource allocation under uncertainty: A deep reinforcement learning-guided optimization approach*. Here each candidate occupies one row in the risk arrays and belongs to a prefecture nested in a province.

## Real-world dataset

[real/real_2025.npz](real/real_2025.npz) contains the allocation inputs for 5,436 Chinese A-share listed firms across 285 prefectures and 31 provinces in 2025. The spatial representation contains 678 undirected adjacency edges. Candidate row numbers serve as identifiers, and the same row order is used in both risk matrices and the retrospective labels.

- `q` contains the 150 joint logistic-regression Bootstrap scenarios used for allocation.
- `q_hgb` contains the 150 joint histogram-gradient-boosting Bootstrap scenarios used for independent evaluation.
- `y2025` records whether each candidate had a qualifying disclosure in 2025. The 223 positive labels are for retrospective evaluation.
- `budgets` lists the five study budgets (109, 163, 272, 544, 815). The scalar `K` is set to 272 for direct use with the HUSAR-Net solver.

The prefecture and province weights are the precomputed values used in the study. [The dataset metadata](real/metadata.json) records counts, file size, and the SHA-256 checksum.

## Synthetic datasets

[synthetic/instances.csv](synthetic/instances.csv) indexes 260 synthetic instances with 150 joint scenarios each. It lists the random seed, dimensions, budget, file size, and SHA-256 checksum. The `file` column is relative to `synthetic/`. Each indexed row has a corresponding NPZ file containing the materialized arrays.

| Candidates | Prefectures | Provinces | Budget | Training | Validation | Main test | Encoding test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 60 | 10 | 50 | 60 | 20 | 20 | 0 |
| 5,000 | 260 | 31 | 250 | 40 | 20 | 20 | 0 |
| 10,000 | 400 | 31 | 500 | 0 | 0 | 10 | 10 |
| 20,000 | 400 | 31 | 1,000 | 20 | 20 | 10 | 0 |
| 50,000 | 400 | 31 | 2,500 | 0 | 0 | 10 | 0 |

The 10,000-candidate main test uses seeds 890000–890009. The separate encoding test uses seeds 880000–880009. The synthetic arrays were generated with NumPy 2.2.6 and SciPy 1.18.1. Risk scenarios are stored as `float64`.

## Common NPZ fields

| Field | Shape | Meaning |
| --- | --- | --- |
| `q` | `(N, B)` | Candidate risk. Each column is a joint scenario across the same candidates. |
| `city`, `prov` | `(N,)` | Prefecture and province of each candidate. |
| `city_prov` | `(G,)` | Province of each prefecture. |
| `edges` | `(E, 2)` | Undirected adjacency between prefectures, with one row per edge. |
| `rho` | `(G,)` | Historical prefecture risk used in adjacency weights. |
| `a_city`, `a_prov` | `(G,)`, `(P,)` | Prefecture and province coverage weights. |
| `K` | Scalar | Number of candidates to select. |
| `alpha`, `beta` | Scalar | Objective weights for hierarchical coverage and spatial protection. |

`N` is the candidate count, `B` is the scenario count, `G` and `P` are the prefecture and province counts, and `E` is the edge count. All indices are zero-based. Each synthetic file also carries its `seed` and a `coords` array containing simulated prefecture coordinates. In synthetic files, `rho` and the coverage weights are simulated.

## Load a dataset

From the repository root, load an NPZ file with the HUSAR-Net CLI.

```powershell
husarnet solve data/real/real_2025.npz
husarnet solve data/synthetic/main_test/n1000/seed_880000.npz
```

To use a different study budget, pass the arrays to `Instance` with the intended `K`.

```python
import numpy as np
from husarnet import Instance

fields = ("q", "city", "prov", "city_prov", "edges", "rho", "a_city", "a_prov")
with np.load("data/real/real_2025.npz", allow_pickle=False) as data:
    instance = Instance(**{field: data[field] for field in fields}, K=109)
```
