---
name: CAVE client split_positions
description: CAVE client returns position columns as separate x/y/z columns when split_positions=True, which is now the preferred/default
type: reference
---

In the CAVE ecosystem, `split_positions=True` is the preferred (and increasingly default) option when fetching synapse or annotation tables. This means position columns arrive already split into separate x, y, z columns rather than as a single array-valued column.

**How to apply:** No unpacking helper is needed in trajan. Design position handling assuming separate coordinate columns throughout (consistent with how `filter_by_soma_distance` works).
