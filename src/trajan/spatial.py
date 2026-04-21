from __future__ import annotations

import polars as pl


def pack_position(
    df: pl.DataFrame | pl.LazyFrame,
    col: str,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Pack separate x/y/z columns into a named struct column with x, y, z fields.

    Infers source column names as {col}_x, {col}_y, {col}_z if not given explicitly.
    The source columns are dropped and replaced with a single struct column named col.

    Parameters
    ----------
    df:
        Input DataFrame or LazyFrame. Pandas DataFrames should be converted with
        pl.from_pandas(df) first.
    col:
        Name for the output struct column, and prefix for inferring source column
        names when x/y/z are not given.
    x, y, z:
        Source column names. Default to {col}_x, {col}_y, {col}_z.

    Returns
    -------
    Same type as input with source columns replaced by a struct column named col.
    """
    x_col = x if x is not None else f"{col}_x"
    y_col = y if y is not None else f"{col}_y"
    z_col = z if z is not None else f"{col}_z"

    struct_expr = pl.struct(
        pl.col(x_col).alias("x"),
        pl.col(y_col).alias("y"),
        pl.col(z_col).alias("z"),
    ).alias(col)

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df.with_columns(struct_expr).drop([x_col, y_col, z_col])
    raise TypeError(
        f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}. "
        "Convert pandas DataFrames with pl.from_pandas(df) first."
    )


def pack_all_positions(
    df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Pack all {col}_x / {col}_y / {col}_z triplets found in df into struct columns.

    Scans the schema for columns ending in _x that have matching _y and _z
    counterparts, and calls pack_position for each. Useful for processing raw
    CAVE output where split_positions=True produces multiple position columns.

    Returns the same type as input.
    """
    if isinstance(df, pl.LazyFrame):
        cols = set(df.collect_schema().names())
    elif isinstance(df, pl.DataFrame):
        cols = set(df.columns)
    else:
        raise TypeError(
            f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}. "
            "Convert pandas DataFrames with pl.from_pandas(df) first."
        )
    prefixes = [
        c[:-2]
        for c in cols
        if c.endswith("_x") and f"{c[:-2]}_y" in cols and f"{c[:-2]}_z" in cols
    ]
    for prefix in prefixes:
        df = pack_position(df, prefix)
    return df


def _sq_distance_expr(col_a: str, col_b: str, axes: tuple[str, ...]) -> pl.Expr:
    a, b = pl.col(col_a), pl.col(col_b)
    terms = [(a.struct.field(ax) - b.struct.field(ax)).pow(2) for ax in axes]
    sq_dist = terms[0] if len(terms) == 1 else pl.sum_horizontal(terms)
    return (
        pl.when(a.is_not_null() & b.is_not_null())
        .then(sq_dist)
        .otherwise(pl.lit(float("nan")))
    )


def euclidean_distance(col_a: str, col_b: str) -> pl.Expr:
    """Euclidean (3-D) distance between two position struct columns."""
    return _sq_distance_expr(col_a, col_b, ("x", "y", "z")).sqrt()


def radial_distance(col_a: str, col_b: str) -> pl.Expr:
    """Distance in the xy plane between two position struct columns, ignoring z."""
    return _sq_distance_expr(col_a, col_b, ("x", "y")).sqrt()
