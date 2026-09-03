"""Config flow to set up the OpenAI Whisper Cloud integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import exceptions
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithConfigEntry,
)
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME, CONF_URL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_BASE_URL,
    CONF_FILE_FIELD,
    CONF_PROMPT,
    CONF_PROVIDER_TYPE,
    CONF_TEMPERATURE,
    CONFIG_MINOR_VERSION,
    CONFIG_VERSION,
    DEFAULT_CUSTOM_MODEL,
    DEFAULT_CUSTOM_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    REQUEST_TIMEOUT,
)
from .whisper_provider import PROVIDERS, WhisperProvider, get_provider

_LOGGER = logging.getLogger(__name__)

PROVIDER_SELECTION_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PROVIDER_TYPE): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=provider.key, label=provider.name)
                    for provider in PROVIDERS.values()
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)

TEMPERATURE_SCHEMA = vol.All(vol.Coerce(float), vol.Range(min=0, max=1))


class CannotConnect(exceptions.HomeAssistantError):
    """Error to indicate we cannot connect to the provider."""


class InvalidAPIKey(exceptions.HomeAssistantError):
    """Error to indicate the API key is invalid."""


class UnauthorizedError(exceptions.HomeAssistantError):
    """Error to indicate the API key lacks the required permissions."""


class WhisperModelNotFound(exceptions.HomeAssistantError):
    """Error to indicate the selected model does not exist."""


async def validate_builtin_provider(
    hass: HomeAssistant, provider: WhisperProvider, model_name: str, api_key: str
) -> None:
    """Verify the API key and the selected model exist on a builtin provider."""
    session = async_get_clientsession(hass)
    url = f"{provider.base_url.rstrip('/')}/v1/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = await session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
    except (TimeoutError, aiohttp.ClientError) as err:
        _LOGGER.warning("Connection test to %s failed: %s", url, err)
        raise CannotConnect from err

    _LOGGER.debug(
        "Model validation request to %s returned %d - %s",
        url,
        response.status,
        response.reason or "",
    )
    if response.status == 401:
        raise InvalidAPIKey
    if response.status == 403:
        raise UnauthorizedError
    if response.status == 404:
        raise WhisperModelNotFound
    if response.status != 200:
        raise CannotConnect


async def validate_custom_provider(
    hass: HomeAssistant, base_url: str, api_key: str
) -> None:
    """Perform a lenient connectivity check against a custom endpoint.

    Many self hosted servers do not implement the models endpoint, so any
    response other than an authentication failure or a connection error is
    accepted. Only clearly broken configurations are rejected.
    """
    session = async_get_clientsession(hass)
    base = base_url.strip().rstrip("/")
    # If the user already included a version prefix, also try the unversioned
    # models path before giving up.
    candidates = [f"{base}/v1/models"]
    if base.rsplit("/", 1)[-1].startswith("v") and base.rsplit("/", 1)[-1][1:].isdigit():
        candidates.insert(0, f"{base}/models")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    reachable = False
    for url in candidates:
        try:
            response = await session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            )
        except (TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.debug("Connection test to %s failed: %s", url, err)
            continue
        reachable = True
        _LOGGER.debug("Custom endpoint check on %s returned %d", url, response.status)
        if response.status == 401:
            raise InvalidAPIKey
        if response.status != 404:
            # Endpoint answered; accept it without further strictness.
            return

    if not reachable:
        raise CannotConnect
    # Every candidate 404'd: the server is reachable, just has no models
    # endpoint. Accept it and let the transcription endpoint fail loudly at
    # runtime if it is also wrong.


def _builtin_schema(provider: WhisperProvider, api_key_optional: bool) -> vol.Schema:
    """Build the form schema for a builtin provider."""
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=f"{provider.name} Whisper"): cv.string,
            vol.Required(CONF_API_KEY) if not api_key_optional else vol.Optional(CONF_API_KEY): cv.string,
            vol.Required(
                CONF_MODEL,
                default=provider.default_model,
            ): vol.In([model.name for model in provider.models]),
            vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): TEMPERATURE_SCHEMA,
            vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        }
    )


CUSTOM_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_CUSTOM_NAME): cv.string,
        vol.Required(CONF_URL): cv.string,
        vol.Optional(CONF_API_KEY): cv.string,
        vol.Required(CONF_MODEL, default=DEFAULT_CUSTOM_MODEL): cv.string,
        vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): TEMPERATURE_SCHEMA,
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        vol.Optional(CONF_FILE_FIELD, default="file"): cv.string,
    }
)


class OptionsFlowHandler(OptionsFlowWithConfigEntry):
    """Handle Whisper Cloud options."""

    config_entry: ConfigEntry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the provider options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        provider = get_provider(self.config_entry.data[CONF_PROVIDER_TYPE])
        current_model = self.config_entry.options.get(
            CONF_MODEL, provider.default_model
        )
        schema: dict[Any, Any] = {}
        if provider.custom:
            schema[vol.Required(CONF_MODEL, default=current_model)] = cv.string
        else:
            model_names = [model.name for model in provider.models]
            if current_model not in model_names:
                current_model = provider.default_model
            schema[vol.Required(CONF_MODEL, default=current_model)] = vol.In(
                model_names
            )
        schema[
            vol.Required(
                CONF_TEMPERATURE,
                default=self.config_entry.options.get(
                    CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                ),
            )
        ] = TEMPERATURE_SCHEMA
        schema[
            vol.Optional(
                CONF_PROMPT,
                default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
            )
        ] = cv.string

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the Whisper Cloud UI config flow."""

    VERSION = CONFIG_VERSION
    MINOR_VERSION = CONFIG_MINOR_VERSION

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._provider: WhisperProvider | None = None

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlowHandler:
        """Create the options flow handler."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the provider selection step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._provider = get_provider(user_input[CONF_PROVIDER_TYPE])
            if self._provider.custom:
                return await self.async_step_custom()
            return await self.async_step_whisper()

        return self.async_show_form(
            step_id="user", data_schema=PROVIDER_SELECTION_SCHEMA, errors=errors
        )

    async def async_step_whisper(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle setup of a builtin provider."""
        assert self._provider is not None
        provider = self._provider
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await validate_builtin_provider(
                    self.hass,
                    provider,
                    user_input[CONF_MODEL],
                    user_input.get(CONF_API_KEY, ""),
                )
            except CannotConnect:
                errors["base"] = "connection_error"
            except InvalidAPIKey:
                errors[CONF_API_KEY] = "invalid_api_key"
            except UnauthorizedError:
                errors["base"] = "unauthorized"
            except WhisperModelNotFound:
                errors["base"] = "whisper_not_found"
            else:
                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data={
                        CONF_PROVIDER_TYPE: provider.key,
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_API_KEY: user_input.get(CONF_API_KEY, ""),
                    },
                    options={
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )

        return self.async_show_form(
            step_id="whisper", data_schema=_builtin_schema(provider, False), errors=errors
        )

    async def async_step_custom(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle setup of a custom OpenAI-compatible endpoint."""
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_URL].strip().rstrip("/")
            try:
                cv.url(user_input[CONF_URL])
                await validate_custom_provider(
                    self.hass, base_url, user_input.get(CONF_API_KEY, "")
                )
            except vol.Invalid:
                errors[CONF_URL] = "invalid_url"
            except CannotConnect:
                errors["base"] = "connection_error"
            except InvalidAPIKey:
                errors[CONF_API_KEY] = "invalid_api_key"
            else:
                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data={
                        CONF_PROVIDER_TYPE: "custom",
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_BASE_URL: base_url,
                        CONF_API_KEY: user_input.get(CONF_API_KEY, ""),
                        CONF_FILE_FIELD: user_input.get(CONF_FILE_FIELD, "file"),
                    },
                    options={
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )

        return self.async_show_form(step_id="custom", data_schema=CUSTOM_SCHEMA, errors=errors)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of an existing entry."""
        errors: dict[str, str] = {}
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        assert entry is not None
        provider = get_provider(entry.data[CONF_PROVIDER_TYPE])

        if user_input is not None:
            try:
                api_key = (
                    user_input.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY, "")
                )
                if provider.custom:
                    base_url = user_input[CONF_URL].strip().rstrip("/")
                    cv.url(user_input[CONF_URL])
                    await validate_custom_provider(self.hass, base_url, api_key)
                    new_data: dict[str, Any] = {
                        CONF_PROVIDER_TYPE: provider.key,
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_BASE_URL: base_url,
                        CONF_API_KEY: api_key,
                        CONF_FILE_FIELD: user_input.get(
                            CONF_FILE_FIELD,
                            entry.data.get(CONF_FILE_FIELD, "file"),
                        ),
                    }
                else:
                    await validate_builtin_provider(
                        self.hass, provider, user_input[CONF_MODEL], api_key
                    )
                    new_data = {
                        CONF_PROVIDER_TYPE: provider.key,
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_API_KEY: api_key,
                    }
            except vol.Invalid:
                errors[CONF_URL] = "invalid_url"
            except CannotConnect:
                errors["base"] = "connection_error"
            except InvalidAPIKey:
                errors[CONF_API_KEY] = "invalid_api_key"
            except UnauthorizedError:
                errors["base"] = "unauthorized"
            except WhisperModelNotFound:
                errors["base"] = "whisper_not_found"
            else:
                # The update listener set up in async_setup_entry reloads the
                # entry automatically.
                self.hass.config_entries.async_update_entry(
                    entry,
                    title=user_input[CONF_NAME],
                    data=new_data,
                    options={
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )
                return self.async_abort(reason="reconfigure_successful")

        if provider.custom:
            schema = vol.Schema(
                {
                    vol.Required(
                        CONF_NAME, default=entry.data.get(CONF_NAME, DEFAULT_CUSTOM_NAME)
                    ): cv.string,
                    vol.Required(CONF_URL): cv.string,
                    vol.Optional(CONF_API_KEY): cv.string,
                    vol.Required(CONF_MODEL): cv.string,
                    vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): TEMPERATURE_SCHEMA,
                    vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
                    vol.Optional(CONF_FILE_FIELD): cv.string,
                }
            )
            suggested = {
                CONF_NAME: entry.data.get(CONF_NAME),
                CONF_URL: entry.data.get(CONF_BASE_URL),
                CONF_MODEL: entry.options.get(CONF_MODEL),
                CONF_TEMPERATURE: entry.options.get(CONF_TEMPERATURE),
                CONF_PROMPT: entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                CONF_FILE_FIELD: entry.data.get(CONF_FILE_FIELD, "file"),
            }
        else:
            schema = _builtin_schema(provider, api_key_optional=True)
            suggested = {
                CONF_NAME: entry.data.get(CONF_NAME),
                CONF_MODEL: entry.options.get(CONF_MODEL, provider.default_model),
                CONF_TEMPERATURE: entry.options.get(CONF_TEMPERATURE),
                CONF_PROMPT: entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
            }

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                data_schema=schema, suggested_values=suggested
            ),
            errors=errors,
        )
