"""Tests for config entry migration and lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from homeassistant.config_entries import ConfigEntryState

from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.openai_whisper_cloud.const import (
    CONF_BASE_URL,
    CONF_PROMPT,
    CONF_PROVIDER_TYPE,
    CONF_STREAMING,
    CONF_TEMPERATURE,
    CONFIG_MINOR_VERSION,
    CONFIG_VERSION,
    DEFAULT_PROMPT,
    DEFAULT_STREAMING,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)


async def setup_legacy_entry(hass: Any, entry: MockConfigEntry) -> MockConfigEntry:
    """Add a legacy entry and run setup (which triggers migration)."""
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_current_format_entry_only_bumps_version(hass: Any) -> None:
    """Entries already in the current format are only version-bumped."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION,
        minor_version=0,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "N", CONF_API_KEY: "k"},
        options={CONF_MODEL: "whisper-1", CONF_TEMPERATURE: 0.3, CONF_PROMPT: "p"},
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.version == CONFIG_VERSION
    assert entry.minor_version == CONFIG_MINOR_VERSION
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("stt.n") is not None


async def test_newer_version_cannot_be_downgraded(hass: Any) -> None:
    """Entries from a newer integration version refuse to migrate."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION + 1,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "N", CONF_API_KEY: "k"},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.state is ConfigEntryState.MIGRATION_ERROR


@pytest.mark.parametrize(
    ("source", "expected_provider"),
    [
        (0, "openai"),
        (1, "groqcloud"),
        (2, "mistral"),
        (3, "cohere"),
        (99, "openai"),  # unknown index falls back
        ("0", "openai"),  # non-integer falls back
    ],
)
async def test_migrate_legacy_source_index(
    hass: Any, source: Any, expected_provider: str
) -> None:
    """The legacy source index becomes a stable provider key."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=1,
        data={"source": source, CONF_NAME: "Legacy", CONF_API_KEY: "k"},
        options={CONF_MODEL: 1, CONF_TEMPERATURE: 0.1, CONF_PROMPT: "hi"},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.state is ConfigEntryState.LOADED
    assert entry.data[CONF_PROVIDER_TYPE] == expected_provider
    # the model index was resolved to a stable name
    assert isinstance(entry.options[CONF_MODEL], str)
    assert entry.options[CONF_TEMPERATURE] == 0.1
    assert entry.options[CONF_PROMPT] == "hi"
    # migration leaves new options to their defaults; the entity applies them
    assert CONF_STREAMING not in entry.options


async def test_migrate_legacy_model_index_out_of_range(hass: Any) -> None:
    """An out-of-range model index resolves to the provider default."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=1,
        data={"source": 0, CONF_NAME: "Legacy", CONF_API_KEY: "k"},
        options={CONF_MODEL: 99, CONF_TEMPERATURE: 0.1, CONF_PROMPT: "hi"},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.data[CONF_PROVIDER_TYPE] == "openai"
    assert entry.options[CONF_MODEL] == "gpt-4o-mini-transcribe"


async def test_migrate_legacy_custom_provider(hass: Any) -> None:
    """Legacy custom entries keep URL, key and model name."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=1,
        data={
            "custom_provider": True,
            "url": "http://127.0.0.1:8000/asr/",
            CONF_NAME: "Legacy",
            CONF_API_KEY: "k",
            CONF_MODEL: "legacy-model",
        },
        options={},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.state is ConfigEntryState.LOADED
    assert entry.data[CONF_PROVIDER_TYPE] == "custom"
    assert entry.data[CONF_BASE_URL] == "http://127.0.0.1:8000/asr/"
    assert entry.data[CONF_API_KEY] == "k"
    assert entry.data[CONF_NAME] == "Legacy"
    assert entry.options == {
        CONF_MODEL: "legacy-model",
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_PROMPT: DEFAULT_PROMPT,
    }


async def test_migrate_legacy_custom_without_key(hass: Any) -> None:
    """Custom entries without an API key do not gain an empty one."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=1,
        data={
            "custom_provider": True,
            "url": "http://127.0.0.1:8000",
            CONF_NAME: "Legacy",
        },
        options={CONF_MODEL: "m"},
    )
    await setup_legacy_entry(hass, entry)
    assert CONF_API_KEY not in entry.data


async def test_migrate_pre_1_1_settings_stored_in_data(hass: Any) -> None:
    """Very old entries had temperature/prompt in data, not options."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=1,
        data={
            "source": 0,
            CONF_NAME: "Old",
            CONF_API_KEY: "k",
            CONF_TEMPERATURE: 0.7,
            CONF_PROMPT: "living room",
        },
        options={},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.state is ConfigEntryState.LOADED
    assert entry.options[CONF_TEMPERATURE] == 0.7
    assert entry.options[CONF_PROMPT] == "living room"
    # no model in options: index 0 resolves to the first model
    assert entry.options[CONF_MODEL] == "whisper-1"
    assert CONF_TEMPERATURE not in entry.data
    assert CONF_PROMPT not in entry.data


async def test_migrate_legacy_version_too_high(hass: Any) -> None:
    """Legacy entries claiming a newer version refuse to migrate."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=4,
        data={"source": 0, CONF_NAME: "Legacy", CONF_API_KEY: "k"},
    )
    await setup_legacy_entry(hass, entry)
    assert entry.state is ConfigEntryState.MIGRATION_ERROR


async def test_entry_unloads_cleanly(hass: Any) -> None:
    """Unloading removes the entity."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION,
        minor_version=CONFIG_MINOR_VERSION,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "Unload Me", CONF_API_KEY: "k"},
        options={CONF_MODEL: "whisper-1"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("stt.unload_me") is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.NOT_LOADED
    # Home Assistant keeps a restored unavailable state for removed entities
    assert hass.states.get("stt.unload_me").state == "unavailable"


async def test_options_update_reloads_entry(hass: Any) -> None:
    """Changing options reloads the entity with the new settings."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION,
        minor_version=CONFIG_MINOR_VERSION,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "Reloaded", CONF_API_KEY: "k"},
        options={CONF_MODEL: "whisper-1"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("stt.reloaded").attributes["model"] == "whisper-1"

    hass.config_entries.async_update_entry(
        entry, options={CONF_MODEL: "gpt-4o-mini-transcribe"}
    )
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    assert (
        hass.states.get("stt.reloaded").attributes["model"] == "gpt-4o-mini-transcribe"
    )


async def test_migrate_refuses_future_versions_directly(hass: Any) -> None:
    """The migration guards refuse entries from newer releases."""
    from custom_components.openai_whisper_cloud import async_migrate_entry

    # current format, but from a future release of the integration
    future = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION + 1,
        minor_version=0,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "N", CONF_API_KEY: "k"},
    )
    future.add_to_hass(hass)
    assert await async_migrate_entry(hass, future) is False

    # legacy shaped data claiming a future version (historical bug shape)
    legacy = MockConfigEntry(
        domain=DOMAIN,
        version=4,
        data={"source": 0, CONF_NAME: "N", CONF_API_KEY: "k"},
    )
    legacy.add_to_hass(hass)
    assert await async_migrate_entry(hass, legacy) is False
