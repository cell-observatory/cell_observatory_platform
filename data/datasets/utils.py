"""Channel selection helpers for the loader.

The DB hands the loader three positionally-aligned arrays instead of the old
``{idx: role}`` jsonb map, so the parse / digit-vs-string sort / ``\\b``-anchored
regex machinery this module used to carry is gone. Position ``k`` of every
aligned array describes channel ``channel_idx[k]``.
"""

import ujson

from typing import Any, List, Optional, Sequence


def _normalize_channel_token(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _as_list(value: Any) -> Optional[list]:
    """Arrow / numpy / pyarrow scalars all reach the actor differently."""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None:
        return None
    return list(value)


def _parse_channel_mapping(raw: object) -> dict[str, str]:
    """Parse the ``{position -> role}`` table the loader writes into metainfo.

    The table is built by :func:`remap_channel_roles_to_selection` from the row's
    channel_type / annotation_type arrays, then serialized to a JSON string per
    row because Ray batches want a flat column -- hence a parse on the way back
    out.
    """
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): _normalize_channel_token(value) for key, value in raw.items()}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        payload = raw.strip()
        if not payload or payload.lower() == "null":
            return {}
        parsed = ujson.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected a channel role JSON object, got {type(parsed).__name__}"
            )
        return {str(key): _normalize_channel_token(value) for key, value in parsed.items()}
    raise ValueError(
        f"Unsupported channel role payload type {type(raw).__name__}; expected "
        f"dict or JSON string"
    )


def assert_dense_channel_idx(channel_idx: Sequence[int]) -> None:
    """Require ``channel_idx == [0, 1, ..., C-1]``.

    The DB keeps ``channel_idx`` as its own column, so in principle an array
    POSITION and a zarr C-AXIS INDEX are different numbers. In practice they are
    not: every row of both training views carries ``{0,1,2,3,4,5}``, and
    ``roi_channels`` is dense 0..n-1 for every ROI.

    Enforcing that here collapses the two numbering schemes into one, which is
    the point -- code that indexes by position and code that indexes by
    channel_idx are then both right, and cannot silently disagree. If the DB
    ever emits a sparse array this raises instead of reading the wrong channels,
    and the selection paths need a deliberate revisit (both this module and
    preprocessor._split_channels reason in index space).
    """
    expected = list(range(len(channel_idx)))
    actual = [int(v) for v in channel_idx]
    if actual != expected:
        raise ValueError(
            f"channel_idx must be dense and ascending from 0; got {actual}. "
            f"Array position and zarr C-axis index are assumed to be the same "
            f"number throughout the loader and preprocessor -- a sparse "
            f"channel_idx breaks that assumption silently, so it is rejected."
        )


def resolve_channel_indices(
    channel_idx: Sequence[int],
    channel_type: Sequence[str],
    localization: Sequence[Optional[str]],
    requested_localizations: Optional[Sequence[str]],
) -> Optional[List[int]]:
    """Zarr C-axis indices to load: requested DATA channels, then ALL mask channels.

    Four properties hold:

    1. ``channel_idx[k]`` is returned, and :func:`assert_dense_channel_idx`
       guarantees it equals ``k``. The DB models array position and zarr C-axis
       index as separate things; rejecting any row where they would differ lets
       the loader and the preprocessor treat them as one number.
    2. Mask channels are ALWAYS retained. ``roi_channels``' XOR constraint forces
       ``localization`` NULL on a mask channel, so a localization-only selection
       can never name the labelmap: it would be dropped silently and
       ``_split_channels`` would then find no object channels. Selection filters
       SIGNAL channels only.
    3. Masks are appended LAST, which is what ``preprocessor._split_channels``
       requires -- signal channels a contiguous prefix, object channels in the
       tail (it raises otherwise).
    4. Selected data channels keep their SOURCE order regardless of the order the
       localizations were requested in, so channel ``k`` of the emitted tensor is
       a function of the row and the selected set only -- never of YAML list
       order.

    Returns ``None`` when there is nothing to select: load every channel in
    on-disk order.
    """
    channel_idx = _as_list(channel_idx)
    channel_type = _as_list(channel_type)
    localization = _as_list(localization) or []

    if channel_idx is None or channel_type is None:
        # array_agg over an empty set yields NULL, not an empty array: the ROI
        # has no channel rows at all. Loading channel 0 by default would be a
        # guess about data we know nothing about.
        raise ValueError(
            "row has no channel metadata (channel_idx is NULL); the ROI has no "
            "dry_lab.roi_channels rows"
        )
    if len(channel_type) != len(channel_idx):
        raise ValueError(
            f"channel arrays are not aligned: channel_idx has {len(channel_idx)} "
            f"entries, channel_type has {len(channel_type)}"
        )
    assert_dense_channel_idx(channel_idx)

    masks = [
        int(channel_idx[k])
        for k, ctype in enumerate(channel_type)
        if _normalize_channel_token(ctype) == "mask"
    ]

    if not requested_localizations:
        if not masks:
            return None
        # Even with no localization filter we must emit an explicit order, so the
        # mask channels land in the tail rather than wherever the ROI put them.
        data = [
            int(channel_idx[k])
            for k, ctype in enumerate(channel_type)
            if _normalize_channel_token(ctype) != "mask"
        ]
        return data + masks

    roles = [None if v is None else _normalize_channel_token(v) for v in localization]
    wanted = {_normalize_channel_token(w) for w in requested_localizations}

    missing = sorted(
        token for token in wanted
        if not any(
            role == token and _normalize_channel_token(channel_type[k]) != "mask"
            for k, role in enumerate(roles)
        )
    )
    if missing:
        raise ValueError(
            f"localization(s) {missing!r} not present among data channels; row has "
            f"{list(zip(channel_idx, channel_type, roles))!r}"
        )

    # SOURCE order, not requested order. selected_channel_localizations is a SET
    # in every config that uses it -- nobody writing [membrane, cytosol] means
    # "and permute the tensor accordingly" -- so driving channel layout from YAML
    # list order makes two configs requesting the same channels produce different
    # tensors, and silently mismatches a checkpoint trained under the other one.
    # Iterating the row's channels instead makes the layout a function of (row,
    # selected set) alone, and dedupes a repeated token for free.
    data = [
        int(channel_idx[k])
        for k, role in enumerate(roles)
        if role in wanted and _normalize_channel_token(channel_type[k]) != "mask"
    ]
    return data + masks


def remap_channel_roles_to_selection(
    channel_type: Sequence[str],
    annotation_type: Sequence[Optional[str]],
    channel_idx: Sequence[int],
    selected_indices: Optional[Sequence[int]],
) -> dict[str, str]:
    """Roles keyed by POST-selection position.

    After the loader slices channels with ``selected_indices``, channel ``k`` of
    the emitted tensor is source channel ``selected_indices[k]`` -- so every
    downstream consumer of the role table (the preprocessor's role-driven
    partition, the inferencer's provenance) must see roles keyed by the NEW
    positions, not the source ones.
    """
    channel_idx = _as_list(channel_idx) or []
    channel_type = _as_list(channel_type) or []
    annotation_type = _as_list(annotation_type) or []
    assert_dense_channel_idx(channel_idx)

    by_source: dict[int, str] = {}
    for position, source in enumerate(channel_idx):
        if _normalize_channel_token(channel_type[position]) != "mask":
            continue
        nk = _normalize_channel_token(annotation_type[position])
        if not nk or nk == "none":
            raise ValueError(
                f"channel_idx={source} is channel_type='mask' with no annotation_type"
            )
        by_source[int(source)] = f"{nk}_masks"

    if selected_indices is None:
        selected_indices = [int(i) for i in channel_idx]

    remapped: dict[str, str] = {}
    for new_idx, source in enumerate(selected_indices):
        role = by_source.get(int(source))
        if role is not None:
            remapped[str(new_idx)] = role
    return remapped
