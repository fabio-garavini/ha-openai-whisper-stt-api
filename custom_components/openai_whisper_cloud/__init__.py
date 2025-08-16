"""OpenAI Whisper Cloud integration."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_NAME,
    CONF_SOURCE,
    Platform,
)
from homeassistant.core import HomeAssistant

from .const import _LOGGER, CONF_PROMPT, CONF_TEMPERATURE

PLATFORMS = [Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Load entry."""

    _LOGGER.info("Setting up %s", entry.entry_id)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_listener))

    return True


async def async_update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Update entry."""
    await hass.config_entries.async_reload(entry.entry_id)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""

    _LOGGER.info("Unloading %s", entry.entry_id)

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry):
    """Migrate old entry."""
    _LOGGER.info(
        f"Migration of {config_entry.entry_id} from version: {config_entry.version}.{config_entry.minor_version}"
    )

    if config_entry.version > 1:
        # This means the user has downgraded from a future version
        return False

    if config_entry.version == 0:
        new_data = {**config_entry.data}

        new_data[CONF_NAME] = "OpenAI Whisper"

        hass.config_entries.async_update_entry(
            config_entry, data=new_data, minor_version=0, version=1
        )

    if config_entry.version == 1 and config_entry.minor_version == 0:
        new_data = {
            CONF_SOURCE: 0,
            CONF_NAME: config_entry.data[CONF_NAME],
            CONF_API_KEY: config_entry.data[CONF_API_KEY],
        }

        new_options = {
            CONF_MODEL: 0,
            CONF_TEMPERATURE: config_entry.data[CONF_TEMPERATURE],
            CONF_PROMPT: config_entry.data[CONF_PROMPT],
        }

        hass.config_entries.async_update_entry(
            config_entry, data=new_data, options=new_options, minor_version=1, version=1
        )

    if config_entry.version == 1 and config_entry.minor_version == 2 and config_entry.data.get(CONF_SOURCE) == 1 and config_entry.options.get(CONF_MODEL) == 2:
        new_data = {
            CONF_SOURCE: 1,
            CONF_NAME: config_entry.data[CONF_NAME],
            CONF_API_KEY: config_entry.data[CONF_API_KEY],
        }

        new_options = {
            CONF_MODEL: 0,
            CONF_TEMPERATURE: config_entry.options[CONF_TEMPERATURE],
            CONF_PROMPT: config_entry.options[CONF_PROMPT],
        }

        hass.config_entries.async_update_entry(
            config_entry, data=new_data, options=new_options, minor_version=1, version=3
        )

    _LOGGER.info(f"Migration of {config_entry.entry_id} successful")

    return True
