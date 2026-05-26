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
    df : pl.DataFrame or pl.LazyFrame
        Input DataFrame or LazyFrame. Pandas DataFrames should be converted with
        pl.from_pandas(df) first.
    col : str
        Name for the output struct column, and prefix for inferring source column
        names when x/y/z are not given.
    x : str or None, optional
        Source column name for x. Defaults to {col}_x.
    y : str or None, optional
        Source column name for y. Defaults to {col}_y.
    z : str or None, optional
        Source column name for z. Defaults to {col}_z.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Same type as input with source columns replaced by a struct column named col.

    Examples
    --------
    Pack inferred column names ({col}_x/y/z):

    >>> df = pack_position(df, "soma_pt_position")

    Explicit source columns:

    >>> df = pack_position(df, "pos", x="x_nm", y="y_nm", z="z_nm")
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


def unpack_position(
    df: pl.DataFrame | pl.LazyFrame,
    col: str,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Unpack a position struct column into separate x/y/z columns.

    Inverse of :func:`pack_position`. The struct column is dropped and replaced
    with three flat columns named ``{col}_x``, ``{col}_y``, ``{col}_z`` (or the
    explicit names given by x/y/z).

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input DataFrame or LazyFrame.
    col : str
        Name of the struct column to unpack. Must have x, y, z fields.
    x : str or None, optional
        Output column name for x. Defaults to {col}_x.
    y : str or None, optional
        Output column name for y. Defaults to {col}_y.
    z : str or None, optional
        Output column name for z. Defaults to {col}_z.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Same type as input with the struct column replaced by three flat columns.

    Examples
    --------
    >>> df = unpack_position(df, "soma_pt_position")
    >>> df.columns  # soma_pt_position replaced by _x/_y/_z
    ['root_id', 'soma_pt_position_x', 'soma_pt_position_y', 'soma_pt_position_z']

    Explicit output names:

    >>> df = unpack_position(df, "pos", x="x_nm", y="y_nm", z="z_nm")
    """
    x_col = x if x is not None else f"{col}_x"
    y_col = y if y is not None else f"{col}_y"
    z_col = z if z is not None else f"{col}_z"

    src = pl.col(col)
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df.with_columns(
            src.struct.field("x").alias(x_col),
            src.struct.field("y").alias(y_col),
            src.struct.field("z").alias(z_col),
        ).drop(col)
    raise TypeError(
        f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}. "
        "Convert pandas DataFrames with pl.from_pandas(df) first."
    )


def unpack_all_positions(
    df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Unpack all position struct columns with x, y, z fields into flat columns.

    Scans the schema for struct columns whose fields are exactly x, y, z and
    calls :func:`unpack_position` for each. Useful for exporting to formats that
    cannot represent structs (e.g., pandas, CSV).

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input DataFrame or LazyFrame to unpack.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Same type as input with all x/y/z struct columns replaced by flat columns.

    Examples
    --------
    >>> df = unpack_all_positions(st.df)
    """
    if isinstance(df, pl.LazyFrame):
        schema = df.collect_schema()
    elif isinstance(df, pl.DataFrame):
        schema = df.schema
    else:
        raise TypeError(
            f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}. "
            "Convert pandas DataFrames with pl.from_pandas(df) first."
        )
    for col_name, dtype in schema.items():
        if isinstance(dtype, pl.Struct):
            field_names = {f.name for f in dtype.fields}
            if field_names == {"x", "y", "z"}:
                df = unpack_position(df, col_name)
    return df


def pack_all_positions(
    df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Pack all {col}_x / {col}_y / {col}_z triplets found in df into struct columns.

    Scans the schema for columns ending in _x that have matching _y and _z
    counterparts, and calls pack_position for each. Useful for processing raw
    CAVE output where split_positions=True produces multiple position columns.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input DataFrame or LazyFrame to pack.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Same type as input with all detected x/y/z triplets packed into struct columns.

    Examples
    --------
    >>> df = pack_all_positions(df)
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
    """Euclidean (3-D) distance between two position struct columns.

    Both columns must be structs with x, y, z fields (see pack_position).
    Returns NaN where either column is null.

    Parameters
    ----------
    col_a : str
        Name of the first position struct column.
    col_b : str
        Name of the second position struct column.

    Returns
    -------
    pl.Expr
        Expression computing the 3-D Euclidean distance, element-wise. NaN where
        either column is null.

    Examples
    --------
    As a filter expression:

    >>> st.filter(euclidean_distance("soma_pre", "soma_post") <= 50_000)

    As a computed column:

    >>> st.add_expression("soma_dist", euclidean_distance("soma_pre", "soma_post"))
    """
    return _sq_distance_expr(col_a, col_b, ("x", "y", "z")).sqrt()


def radial_distance(col_a: str, col_b: str) -> pl.Expr:
    """Lateral (horizontal) distance between two position struct columns, ignoring depth.

    Computes distance in the xz plane, where y is the cortical depth axis and x/z
    are the horizontal axes. Both columns must be structs with x, y, z fields
    (see pack_position). Returns NaN where either column is null.

    Parameters
    ----------
    col_a : str
        Name of the first position struct column.
    col_b : str
        Name of the second position struct column.

    Returns
    -------
    pl.Expr
        Expression computing the 2-D lateral distance in the xz plane,
        element-wise. NaN where either column is null.

    Examples
    --------
    >>> st.filter(radial_distance("soma_pre", "soma_post") <= 30_000)
    """
    return _sq_distance_expr(col_a, col_b, ("x", "z")).sqrt()


def spatial_feature_exprs(
    from_col: str,
    to_col: str,
    depth_axis: str = "y",
    euclidean: bool = True,
    depth_diff: bool = True,
    spherical: bool = True,
    cylindrical: bool = True,
) -> dict[str, pl.Expr]:
    """Compute a battery of spatial features for the vector from_col → to_col.

    Both columns must be structs with x, y, z fields (see pack_position).
    All returned expressions evaluate to NaN (not null) where either column is null.

    The vector **v = to_pos − from_pos** is decomposed into depth and lateral
    components relative to ``depth_axis``. With ``depth_axis="y"`` (the default
    cortical convention where y increases with depth), the lateral plane is xz
    and ``rho`` equals :func:`radial_distance` for the same pair of columns.

    Parameters
    ----------
    from_col : str
        Name of the origin position struct column.
    to_col : str
        Name of the destination position struct column.
    depth_axis : str, optional
        Axis that represents cortical depth, optionally with a direction suffix.
        Plain ``"x"``, ``"y"``, or ``"z"`` means positive values go deeper.
        Append ``"_r"`` to reverse: ``"y_r"`` means positive y is towards the
        surface (shallower). Defaults to ``"y"``.
    euclidean : bool, optional
        Include the Euclidean (3-D) distance. Key: ``"euclidean"``.
    depth_diff : bool, optional
        Include the signed depth component of the vector. Key: ``"depth_diff"``.
    spherical : bool, optional
        Include spherical coordinates: ``"r"`` (= euclidean), ``"theta"``
        (polar angle from depth axis, [0, π]), ``"phi"`` (azimuthal angle in
        lateral plane, [-π, π]).
    cylindrical : bool, optional
        Include cylindrical coordinates: ``"rho"`` (lateral distance), ``"phi"``
        (shared with spherical), ``"dy"`` (= depth_diff).

    Returns
    -------
    dict[str, pl.Expr]
        Mapping of feature name to Polars expression. ``"phi"`` appears at most
        once even when both ``spherical`` and ``cylindrical`` are True.

    Examples
    --------
    >>> feats = spatial_feature_exprs("soma_pre", "soma_post")
    >>> df.select([expr.alias(name) for name, expr in feats.items()])

    Reverse the depth convention (positive y = shallower):

    >>> feats = spatial_feature_exprs("soma_pre", "soma_post", depth_axis="y_r")
    """
    # Parse optional "_r" suffix — reverses the sign of the depth component
    if depth_axis.endswith("_r"):
        axis = depth_axis[:-2]
        depth_sign = -1
    else:
        axis = depth_axis
        depth_sign = 1

    fc, tc = pl.col(from_col), pl.col(to_col)
    null_cond = fc.is_not_null() & tc.is_not_null()
    nan = pl.lit(float("nan"))

    def _guard(expr: pl.Expr) -> pl.Expr:
        return pl.when(null_cond).then(expr).otherwise(nan)

    # Lateral axes: the two axes not equal to axis, sorted for determinism
    lateral = tuple(sorted({"x", "y", "z"} - {axis}))
    lat_a, lat_b = lateral

    da = tc.struct.field(lat_a) - fc.struct.field(lat_a)
    db = tc.struct.field(lat_b) - fc.struct.field(lat_b)
    dd = depth_sign * (tc.struct.field(axis) - fc.struct.field(axis))

    r_sq = da.pow(2) + db.pow(2) + dd.pow(2)
    r_expr = r_sq.sqrt()

    result: dict[str, pl.Expr] = {}

    if euclidean:
        result["euclidean"] = _guard(r_expr)

    if depth_diff:
        result["depth_diff"] = _guard(dd)

    need_phi = spherical or cylindrical
    phi_expr = pl.arctan2(db, da) if need_phi else None

    if spherical:
        result["r"] = _guard(r_expr)
        result["theta"] = _guard((dd / r_expr).arccos())
        result["phi"] = _guard(phi_expr)

    if cylindrical:
        result["rho"] = _guard((da.pow(2) + db.pow(2)).sqrt())
        if "phi" not in result:
            result["phi"] = _guard(phi_expr)
        result["dy"] = _guard(dd)

    return result
