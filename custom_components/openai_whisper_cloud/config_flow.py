"""Config flow for OpenAI Whisper Cloud integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests
import voluptuous as vol

from homeassistant import exceptions
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME
import homeassistant.helpers.config_validation as cv

from .const import (
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_WHISPER_MODEL,
    DOMAIN,
    SUPPORTED_MODELS,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_MODEL, default=DEFAULT_WHISPER_MODEL): vol.In(
            SUPPORTED_MODELS
        ),
        vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(
            vol.Coerce(float), vol.Range(min=0, max=1)
        ),
        vol.Optional(CONF_PROMPT): cv.string,
    }
)

_LOGGER = logging.getLogger(__name__)


async def validate_input(data: dict):
    """Validate the user input."""

    obscured_api_key = data.get(CONF_API_KEY)
    data[CONF_API_KEY] = "<api_key>"
    _LOGGER.debug("User validation got: %s", data)
    data[CONF_API_KEY] = obscured_api_key

    if data.get(CONF_TEMPERATURE) is None:
        data[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
    if data.get(CONF_PROMPT) is None:
        data[CONF_PROMPT] = DEFAULT_PROMPT

    response = await asyncio.to_thread(
        requests.get,
        url="https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {data.get(CONF_API_KEY)}",
            "Content-Type": "application/json"
        },
    )

    _LOGGER.debug("Models request took %f s and returned %d - %s", response.elapsed.seconds, response.status_code, response.reason)

    if response.status_code == 401:
        raise InvalidAPIKey

    if response.status_code == 403:
        raise UnauthorizedError

    if response.status_code != 200:
        raise UnknownError

    for model in response.json().get("data", []):
        if model.get("id") == data.get(CONF_MODEL):
            break
        if model == response.json().get("data")[-1]:
            raise WhisperModelBlocked
    
    _LOGGER.debug("User validation successful")


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle UI config flow."""

    VERSION = 1
    MINOR_VERSION = 0

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Handle initial step."""
        errors = {}
        if user_input is not None:
            try:
                await validate_input(user_input)

                return self.async_create_entry(
                    title=user_input[CONF_NAME], data=user_input
                )

            except requests.exceptions.RequestException as e:
                _LOGGER.error(e)
                errors["base"] = "connection_error"
            except UnauthorizedError:
                _LOGGER.exception("Unauthorized")
                errors["base"] = "unauthorized"
            except InvalidAPIKey:
                _LOGGER.exception("Invalid API key")
                errors[CONF_API_KEY] = "invalid_api_key"
            except WhisperModelBlocked:
                _LOGGER.exception("Whisper Model Not Found")
                errors["base"] = "whisper_blocked"
            except UnknownError:
                _LOGGER.exception("Unknown error")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )


class UnknownError(exceptions.HomeAssistantError):
    """Unknown error."""


class UnauthorizedError(exceptions.HomeAssistantError):
    """API key valid but doesn't have the rights to use Whisper."""


class InvalidAPIKey(exceptions.HomeAssistantError):
    """Invalid api_key error."""


class WhisperModelBlocked(exceptions.HomeAssistantError):
    """Whisper not found in the available OpenAI models."""
