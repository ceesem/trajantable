# Volume edge correction

A dataset only holds the cells inside its imaged volume. Around a cell near the lateral
edge, part of every ring of radius ρ lies outside the volume, so counting the candidate
partners in that ring undercounts it: the cells there were never imaged, they are not
absent. Any estimate that scales a connection probability by "how many cells of a type
sit at distance ρ" inherits that undercount, and the narrower the volume the worse it is.

`trajan.edge_weight` fixes the *count*, not the probability. Each observed candidate is
weighted by `1 / (fraction of its ring inside the volume)`: a cell found in a ring that is
40 % inside stands for 2.5 cells (Horvitz–Thompson weighting). Connection probabilities
need no correction, since they are ratios among pairs that are all inside.

## Usage

```python
from trajan import counts, edge_weight, lateral_box, possible_pairs, radial_distance, with_distance

box = lateral_box(cells, "soma")                     # 1st–99th percentile footprint
pu = possible_pairs(st).filter_by_radial_distance(400)
pu = with_distance(pu, "d_rho", radial_distance)
pu = edge_weight(pu, box, anchor="post")             # rings around the postsynaptic cell
counts(pu, bin_by={"d_rho": [0, 50, 100, 200, 300, 400]}, group_by="cell_type_pre")
```

The result carries `n_possible` (observed candidate pairs) and `sum_edge_weight`
(edge-corrected candidate pairs) per bin. Use `anchor="post"` when the cell whose
surroundings you count is the target (candidates are presynaptic inputs) and
`anchor="pre"` when it is the source.

## What it rests on

- The volume is approximated by the percentile box of soma positions; percentiles rather
  than min / max because a volume's margins are thinly segmented.
- Cells beyond the edge are assumed to sit at the same density as those inside at the
  same distance. Directional biases (tropism) bend this, mostly in the far tail of
  edge cells.
- The depth axis is not corrected: a cortical volume spans pia to white matter, so the
  fraction depends on the lateral ring alone.
- Weights are capped (default 10×): a ring almost entirely outside is guesswork.
- Cells that are inside the volume but missing from the universe are a different loss,
  measured separately (universe completeness) and not part of this correction.

`ring_inside_fraction` exposes the per-ring fraction as a numpy function for plots and
sanity checks: the corrected counts around an edge cell in a uniform universe should
follow density × ring area, which is what `tests/test_edge_correction.py` asserts.
