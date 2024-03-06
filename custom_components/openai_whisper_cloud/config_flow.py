"""Config flow for OpenAI Whisper Cloud integration."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow
from homeassistant.const import CONF_API_KEY, CONF_MODEL
import homeassistant.helpers.config_validation as cv

from .const import CONF_PROMPT, CONF_TEMPERATURE, DEFAULT_WHISPER_MODEL, DOMAIN

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_MODEL, default=DEFAULT_WHISPER_MODEL): cv.string,
        vol.Required(CONF_TEMPERATURE, default=0): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        vol.Optional(CONF_PROMPT): cv.string,
    }
)


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Whisper Cloud."""

    VERSION = 1

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ):
        """Handle the initial step."""
        if user_input is not None:
            return self.async_create_entry(title="OpenAI Whisper", data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
