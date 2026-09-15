# 09 · Figures (`paper/figures/`)

One `make_fig*.py` generator per figure; each writes both `.pdf` and `.png` into `paper/figures/`. The manuscript (`paper/neurips_2026.tex`) and the `markdown/` deck `\includegraphics` them.

## Render environment (important)

`paper/figures/make_*.py` must run with:

```
/usr/scratch/thomasawalton/envs/protein_env/bin/python paper/figures/make_<fig>.py
```

- `protein_env` has **matplotlib 3.5.3 + numpy + torch**, but **no `transformers`**, and the base conda env has **no matplotlib**. **No single env has matplotlib + torch + transformers together**, and the `alchemab/antiberta2` tokenizer isn't cached locally — so the from-scratch fingerprint compute can't run here.
- Workaround: `make_fig2_fingerprints.py`, `make_fig3_strategies.py`, and `make_fig3_hybrids.py` **defer their `transformers`/`masking` imports into the compute path** and read precomputed caches, so they render with matplotlib only:
  - **`_fingerprints_cache.pkl`** (written by `make_fig3_strategies.py`) = `(static_fps, hybrid_fps, cdr_ranges, L_aa)` — per-strategy mask-probability fingerprints on one representative VH (seed 42, 4000 draws). `make_fig2_fingerprints.py` reads it (cache key `structure` == fig2's `structure_raw`).
  - **`_canonical_region_cache.npz`** — precomputed canonical region boundaries for the schematic figures.
- **mathtext gotcha:** matplotlib 3.5.3 does **not** support `\text{…}`. Use braces for a tight ordinary symbol, e.g. `$\mathtt{structure{-}LR}$`, not `$\mathtt{structure\text{-}LR}$`.

## Unified color scheme

All strategy-colored figures share one palette, anchored on the fingerprint figure (`make_fig2_fingerprints.py`). Use these hex codes for any new figure:

| strategy | hex | | strategy | hex |
|---|---|---|---|---|
| uniform | `#777777` | | germline | `#E45756` |
| cdr | `#F58518` | | intersection | `#DAA520` |
| span | `#54A24B` | | structure | `#B279A2` |
| interface | `#4C78A8` | | structure-LR | `#7B3F99` |

The CDR/framework "chain anatomy" colors in the schematic figures (orange CDR fills, `#FFD27A` CDR shading on fingerprints) are a separate vocabulary and are intentionally **not** strategy colors.

## Key figures

| Generator | Output(s) | What it shows |
|-----------|-----------|---------------|
| `make_fig2_fingerprints.py` | `fig2_fingerprints` | 2×4 panels: empirical per-position mask probability for `uniform, cdr, span, interface, germline, intersection, structure, structure-LR` on one real VH (CDRs shaded). Reads `_fingerprints_cache.pkl`. |
| `make_fig3_hybrids.py` | `fig3_hybrids`, `fig3_hybrids_noranks` | Rank-consistency: each of 14 strategies as a dot/diamond spread of its per-metric ranks (specialists vs hybrids). `_noranks` drops the `#1/#2` callouts; y-axis "rank". |
| `make_fig_radar.py` | `fig_radar`, `fig_radar_legend` | Overlaid radar of the 8 specialists across the 6 metrics. |
| `make_fig_radar_alts.py` | `fig_radar_facets`, `fig_radar_heatmap` | Small-multiple radars (one per strategy, others greyed) and a strategy×task heatmap. Facet order: `uniform, cdr, span, interface, germline, intersection, structure, structure-LR`. **Two data sources:** facets read `seed_aggregate.csv` (3-seed means, matching Tables 1–2); the heatmap still reads seed 42 from `comparison_table.csv`. |
| `make_fig3_strategies.py` | `fig_app_strategies` | Appendix 4×4 roster of all 16 masking strategies as fingerprints. **Writes `_fingerprints_cache.pkl`.** |
| `make_fig_strategy_schematic.py` | `fig_strategy_schematic` + 7 assets | Conceptual schematic of where each strategy spends its budget on a generic VH. Also emits standalone, identically-sized, transparent assets (`_chain`, `_uniform`, `_cdr`, `_interface`, `_span`, `_structure`, `_axis`) for Illustrator composition. |

The six radar metrics (Para, Cont, Struc, Dev, Mut, CDR3) and their exact CSV columns are defined in [07-evaluation-and-metrics](07-evaluation-and-metrics.md). `make_fig_radar_alts.py` plots five of them (no Mut), one per task, so no task is geometrically double-counted.

**Replication caveat for the facets:** they draw seed means with no error bars, and per-spoke min-max normalization stretches the plotted arms to fill [0,1] no matter how small the true range. Two spokes are near-ties — `germline` leads `interface` on contact LR-P@L by 0.001 and on the structure probe by 0.004, against between-seed sds of 0.03–0.06 and 0.01–0.02 — so touching the rim there is not a win. `structure` is now the worst arm on Para, Struc and CDR3 simultaneously, so its facet collapses to a single Dev spike.

The strategy schematic's per-asset files are sized to register when stacked (shared `xlim`, axes fill the canvas, **no** `bbox_inches="tight"`).

## Other generators (less central / not all inspected here)

`make_fig1_data.py` (`fig1_data`), `make_fig1_overview.py` (`fig1_overview`), `make_fig2_specialists.py` (`fig2_specialists`), `make_fig3_experiments.py` (`fig3_experiments`), `make_fig3_knn_distance.py` (`fig3_knn`), `make_fig4_bump.py` (`fig4_bump`), `make_fig5_slope.py` (`fig5_slope`), `make_fig_canonical_regions.py`, `make_fig_canonical_schematic.py`, `make_fig_region_schematics.py`. Most read the comparison table or the caches; confirm a generator's data source before regenerating.

## Regenerating

```bash
P=/usr/scratch/thomasawalton/envs/protein_env/bin/python
$P paper/figures/make_fig2_fingerprints.py     # fingerprints (from cache)
$P paper/figures/make_fig_radar_alts.py         # radar facets (seed means) + heatmap (seed 42)
$P paper/figures/make_fig3_hybrids.py           # hybrids + noranks
$P paper/figures/make_fig_strategy_schematic.py # schematic + assets
```

If the underlying results change, delete `_fingerprints_cache.pkl` so `make_fig3_strategies.py` recomputes it — but recomputation needs an env with `transformers` + the `alchemab/antiberta2` tokenizer, which `protein_env` lacks.
