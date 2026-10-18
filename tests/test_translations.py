"""Tests guarding the translations against ICU message format rules.

The Home Assistant frontend compiles every translated string with
IntlMessageFormat (ICU message syntax). Raw braces in a translation - e.g.
a Home Assistant template example like ``{{ states('sensor.x') }}`` - are
parsed as ICU arguments and break the frontend rendering with
"Translation error: MALFORMED_ARGUMENT".

Literal braces must be ICU-quoted with single quotes, and any apostrophe
inside a quoted block must be doubled::

    raw:     {{ states('sensor.x') }}
    escaped: '{{ states(''sensor.x'') }}'

The scanner below mirrors the frontend's parser for the failure classes
that matter: raw braces outside quoted blocks and malformed arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPONENT_DIR = (
    Path(__file__).parent.parent / "custom_components" / "openai_whisper_cloud"
)
TRANSLATION_FILES = [
    COMPONENT_DIR / "strings.json",
    COMPONENT_DIR / "translations" / "en.json",
    COMPONENT_DIR / "translations" / "it.json",
    COMPONENT_DIR / "translations" / "lv.json",
]


def assert_icu_parseable(message: str) -> None:
    """Assert the message is valid ICU syntax for the HA frontend.

    Rules mirrored from the frontend's parser:
    - outside a quoted block, ``{`` must open a valid ``{name}``
      argument (letters, digits, underscore) and ``}`` must close one;
    - a single quote toggles ICU literal mode; a doubled apostrophe
      ``''`` is a literal apostrophe inside a quoted block.
    """
    in_quote = False
    i = 0
    n = len(message)
    while i < n:
        char = message[i]
        if char == "'":
            if in_quote and i + 1 < n and message[i + 1] == "'":
                i += 2  # literal apostrophe inside a quoted block
                continue
            in_quote = not in_quote
            i += 1
            continue
        if in_quote:
            i += 1
            continue
        if char == "{":
            end = i + 1
            while end < n and (message[end].isalnum() or message[end] == "_"):
                end += 1
            if end == i + 1 or end >= n or message[end] != "}":
                msg = f"malformed ICU argument at position {i}: {message!r}"
                raise AssertionError(msg)
            i = end + 1
            continue
        if char == "}":
            msg = f"unbalanced '}}' at position {i}: {message!r}"
            raise AssertionError(msg)
        i += 1


def walk_strings(data: object, path: str = "") -> list[tuple[str, str]]:
    """Yield (dotted key, value) for every leaf string."""
    out: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            out.extend(walk_strings(value, f"{path}.{key}"))
    else:
        assert isinstance(data, str), f"non-string value at {path}"
        out.append((path, data))
    return out


def test_all_translations_parse_as_icu() -> None:
    """Every translated string must survive the frontend's ICU parser."""
    checked = 0
    for file_path in TRANSLATION_FILES:
        strings = json.loads(file_path.read_text(encoding="utf-8"))
        for key, value in walk_strings(strings):
            assert_icu_parseable(value)
            checked += 1
    assert checked > 200


def test_template_example_renders_verbatim() -> None:
    """The escaped template example renders as the literal Jinja text."""
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text(encoding="utf-8"))
    description = strings["config"]["step"]["whisper"]["data_description"]["prompt"]
    escaped = (
        "'{{ states(''sensor.room_names'') }}, {{ states(''sensor.entity_names'') }}'"
    )
    assert escaped in description
    # un-quote the ICU block and the intended example text appears verbatim
    rendered = description.replace(escaped, "RENDERED")
    assert "RENDERED" in rendered


def test_translations_have_matching_keys() -> None:
    """strings.json and the translations must stay in sync."""

    def leaf_keys(data: object, prefix: tuple[str, ...] = ()) -> set[tuple]:
        keys: set[tuple] = set()
        if isinstance(data, dict):
            for key, value in data.items():
                keys |= leaf_keys(value, (*prefix, key))
        else:
            keys.add(prefix)
        return keys

    source = leaf_keys(
        json.loads((COMPONENT_DIR / "strings.json").read_text(encoding="utf-8"))
    )
    assert source
    for path in TRANSLATION_FILES[1:]:
        translated = leaf_keys(json.loads(path.read_text(encoding="utf-8")))
        assert not source - translated, f"{path}: missing keys {source - translated}"
        assert not translated - source, f"{path}: extra keys {translated - source}"
