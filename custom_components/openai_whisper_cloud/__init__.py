"""The OpenAI Whisper Cloud integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME, Platform
from homeassistant.core import HomeAssistant

from .const import (
    CONF_BASE_URL,
    CONF_PROMPT,
    CONF_PROVIDER_TYPE,
    CONF_TEMPERATURE,
    CONFIG_MINOR_VERSION,
    CONFIG_VERSION,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from .whisper_provider import LEGACY_SOURCE_KEYS, PROVIDERS

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Whisper Cloud from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_listener))
    return True


async def async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when it is updated."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries to the current format.

    The new format stores the provider as a stable string key instead of a
    positional index and always stores the model by name, so adding providers
    or models can never corrupt existing entries.

    Detection of legacy entries is done by the *shape* of the stored data
    rather than the version number, because a bug in the historical v1.3
    migration could bump the stored version to 3 while keeping legacy data.
    """
    _LOGGER.info(
        "Migrating config entry %s from version %s.%s",
        entry.entry_id,
        entry.version,
        entry.minor_version,
    )

    if CONF_PROVIDER_TYPE in entry.data:
        # Already in the current format; only downgrade from a future version
        # is unsupported.
        if entry.version > CONFIG_VERSION:
            _LOGGER.error(
                "Config entry %s was created with a newer version of the integration and cannot be downgraded",
                entry.entry_id,
            )
            return False
        hass.config_entries.async_update_entry(
            entry, version=CONFIG_VERSION, minor_version=CONFIG_MINOR_VERSION
        )
        return True

    # Legacy entries; version 3 may still carry legacy data (historical bug).
    if entry.version > 3:
        _LOGGER.error(
            "Config entry %s was created with a newer version of the integration and cannot be downgraded",
            entry.entry_id,
        )
        return False

    data, options = _migrate_legacy_data(dict(entry.data), dict(entry.options))

    hass.config_entries.async_update_entry(
        entry,
        data=data,
        options=options,
        version=CONFIG_VERSION,
        minor_version=CONFIG_MINOR_VERSION,
    )
    _LOGGER.info("Migration of config entry %s successful", entry.entry_id)
    return True


def _migrate_legacy_data(
    data: dict[str, Any], options: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert legacy entry data/options to the v2 format."""
    data = dict(data)
    options = dict(options)
    data.setdefault(CONF_NAME, DEFAULT_NAME)
    base_url = ""

    if data.pop("custom_provider", False):
        provider_key = "custom"
        base_url = data.pop("url", "")
        # Custom entries always stored the model name (as a string).
        model = str(options.get(CONF_MODEL) or data.pop(CONF_MODEL, "") or "")
    else:
        source = data.pop("source", 0)
        try:
            provider = PROVIDERS[LEGACY_SOURCE_KEYS[int(source)]]
        except (IndexError, ValueError, KeyError):
            provider = PROVIDERS["openai"]
        provider_key = provider.key

        # Pre-1.1 entries kept temperature/prompt (and no model) in data.
        if CONF_TEMPERATURE in data and CONF_TEMPERATURE not in options:
            options[CONF_TEMPERATURE] = data.pop(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            options[CONF_PROMPT] = data.pop(CONF_PROMPT, DEFAULT_PROMPT)
            options.setdefault(CONF_MODEL, 0)

        model = options.get(CONF_MODEL)
        if isinstance(model, int):
            # Legacy entries stored the model as an index into the provider
            # model list; resolve it to a stable name.
            model = (
                provider.models[model].name
                if 0 <= model < len(provider.models)
                else provider.default_model
            )
        model = str(model or provider.default_model)

    new_data: dict[str, Any] = {
        CONF_PROVIDER_TYPE: provider_key,
        CONF_NAME: data[CONF_NAME],
    }
    if data.get(CONF_API_KEY):
        new_data[CONF_API_KEY] = data[CONF_API_KEY]
    if provider_key == "custom":
        new_data[CONF_BASE_URL] = base_url

    new_options: dict[str, Any] = {
        CONF_MODEL: model,
        CONF_TEMPERATURE: float(options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
        CONF_PROMPT: str(options.get(CONF_PROMPT, DEFAULT_PROMPT) or DEFAULT_PROMPT),
    }
    return new_data, new_options
