"""Config flow for OpenAI Whisper Cloud integration."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_MODEL
import homeassistant.helpers.config_validation as cv

from .const import (
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_WHISPER_MODEL,
    DOMAIN,
    SUPPORT_MODELS,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_MODEL, default=DEFAULT_WHISPER_MODEL): vol.In(SUPPORT_MODELS),
        vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        vol.Optional(CONF_PROMPT): cv.string,
    }
)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Whisper Cloud."""

    VERSION = 1

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step."""
        if user_input is not None:
            if user_input.get(CONF_TEMPERATURE) is None:
                user_input[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
            if user_input.get(CONF_PROMPT) is None:
                user_input[CONF_PROMPT] = DEFAULT_PROMPT
            return self.async_create_entry(title="OpenAI Whisper", data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
