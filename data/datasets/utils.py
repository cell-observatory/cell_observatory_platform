import re
from typing import List, Optional

import ujson


def _normalize_channel_token(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _parse_channel_mapping(raw: object) -> dict[str, str]:
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): _normalize_channel_token(value) for key, value in raw.items()}
    if isinstance(raw, str):
        payload = raw.strip()
        if not payload or payload.lower() == "null":
            return {}
        parsed = ujson.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected channel_mapping JSON object, got {type(parsed).__name__}")
        return {str(key): _normalize_channel_token(value) for key, value in parsed.items()}
    raise ValueError(f"Unsupported channel_mapping payload type {type(raw).__name__}; expected dict or JSON string")


def resolve_channel_localization_indices(
    channel_mapping: object,
    requested_localizations: Optional[List[str]],
) -> Optional[List[int]]:
    if not requested_localizations:
        return None

    mapping = _parse_channel_mapping(channel_mapping)
    if not mapping:
        raise ValueError(
            f"channel_mapping is required to resolve selected channel localizations {requested_localizations}"
        )

    def sort_key(item):
        key = str(item[0])
        return (0, int(key)) if key.isdigit() else (1, key)

    ordered = sorted(mapping.items(), key=sort_key)
    resolved: List[int] = []
    for localization in requested_localizations:
        token = _normalize_channel_token(localization)
        match_key = next(
            (
                key
                for key, value in ordered
                if token == value
            ),
            None,
        )
        if match_key is None:
            raise ValueError(
                f"Unable to resolve requested channel localization {localization!r} "
                f"from channel_mapping={mapping!r}"
            )
        resolved.append(int(match_key))
    return resolved