"""Tests for the config, options and reconfigure flows."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME, CONF_URL
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.openai_whisper_cloud.config_flow import (
    CannotConnect,
    InvalidAPIKey,
    UnauthorizedError,
    WhisperModelNotFound,
    validate_custom_provider,
    validate_prompt,
)
from custom_components.openai_whisper_cloud.const import (
    CONF_BASE_URL,
    CONF_FILE_FIELD,
    CONF_PROMPT,
    CONF_PROVIDER_TYPE,
    CONF_STREAMING,
    CONF_TEMPERATURE,
    DEFAULT_PROMPT,
    DEFAULT_STREAMING,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from custom_components.openai_whisper_cloud.whisper_provider import get_provider

VALID_TEMPLATE = "{{ states('sensor.room_names') }}, garage"


def schema_fields(result: dict[str, Any]) -> set[str]:
    """Return the field names of a form's data schema."""
    return {marker.schema for marker in result["data_schema"].schema}


def get_field(result: dict[str, Any], name: str) -> tuple[Any, Any]:
    """Return (marker, validator) of a schema field."""
    for marker, validator in result["data_schema"].schema.items():
        if marker.schema == name:
            return marker, validator
    raise KeyError(name)


# These tests talk to scripted local HTTP servers; the plugin restricts
# real sockets to 127.0.0.1.
pytestmark = pytest.mark.usefixtures("socket_enabled")


# ---------------------------------------------------------------------------
# prompt validation helper
# ---------------------------------------------------------------------------


async def test_validate_prompt_plain(hass: Any) -> None:
    """Plain prompts and empty prompts are valid."""
    assert validate_prompt(hass, "") is None
    assert validate_prompt(hass, "kitchen, garage") is None
    assert validate_prompt(hass, "a { b") is None


async def test_validate_prompt_valid_template(hass: Any) -> None:
    """Syntactically valid templates pass."""
    assert validate_prompt(hass, VALID_TEMPLATE) is None
    assert validate_prompt(hass, "{{ sensor.x }}, {{ states('a') }}") is None


async def test_validate_prompt_invalid_template(hass: Any) -> None:
    """Broken templates return the error key and the engine detail."""
    result = validate_prompt(hass, "{{ /home/user/x.json }}")
    assert result is not None
    key, detail = result
    assert key == "invalid_prompt_template"
    assert "TemplateSyntaxError" in detail

    key, detail = validate_prompt(hass, "{% if")
    assert key == "invalid_prompt_template"
    assert detail


# ---------------------------------------------------------------------------
# builtin provider validation
# ---------------------------------------------------------------------------


async def test_validate_builtin_provider_single_model(
    hass: Any, transcription_server_factory: Any
) -> None:
    """Providers with a retrieve endpoint are checked per model."""
    from custom_components.openai_whisper_cloud.config_flow import (
        validate_builtin_provider,
    )

    server = await transcription_server_factory(
        [], models_responses=[(200, {"id": "whisper-1", "object": "model"})]
    )
    provider = dataclasses.replace(get_provider("openai"), base_url=server.url)
    await validate_builtin_provider(hass, provider, "whisper-1", "key")
    assert server.models_requests[0]["path"].endswith("/v1/models/whisper-1")
    assert server.models_requests[0]["auth"] == "Bearer key"


async def test_validate_builtin_provider_statuses(
    hass: Any, transcription_server_factory: Any
) -> None:
    """HTTP statuses map to the right setup errors."""
    from custom_components.openai_whisper_cloud.config_flow import (
        validate_builtin_provider,
    )

    server = await transcription_server_factory(
        [], models_responses=[(401, "bad"), (403, "no"), (404, "gone"), (500, "oops")]
    )
    provider = dataclasses.replace(get_provider("openai"), base_url=server.url)
    with pytest.raises(InvalidAPIKey):
        await validate_builtin_provider(hass, provider, "whisper-1", "key")
    with pytest.raises(UnauthorizedError):
        await validate_builtin_provider(hass, provider, "whisper-1", "key")
    with pytest.raises(WhisperModelNotFound):
        await validate_builtin_provider(hass, provider, "whisper-1", "key")
    with pytest.raises(CannotConnect):
        await validate_builtin_provider(hass, provider, "whisper-1", "key")


async def test_validate_builtin_provider_unreachable(
    hass: Any, transcription_server_factory: Any
) -> None:
    """A dead endpoint is a connection error."""
    from custom_components.openai_whisper_cloud.config_flow import (
        validate_builtin_provider,
    )

    server = await transcription_server_factory([])
    url = server.url
    await server.stop()
    provider = dataclasses.replace(get_provider("openai"), base_url=url)
    with pytest.raises(CannotConnect):
        await validate_builtin_provider(hass, provider, "whisper-1", "key")


async def test_validate_builtin_provider_model_list(
    hass: Any, transcription_server_factory: Any
) -> None:
    """List-only providers (MiMo) are checked against the model list."""
    from custom_components.openai_whisper_cloud.config_flow import (
        validate_builtin_provider,
    )

    server = await transcription_server_factory(
        [],
        models_responses=[
            (200, {"object": "list", "data": [{"id": "mimo-v2.5-asr"}]}),
            (200, {"object": "list", "data": [{"id": "other"}]}),
            (200, "not json at all"),
            (200, {"unexpected": "shape"}),
            (404, "no list"),
        ],
    )
    provider = dataclasses.replace(get_provider("xiaomimimo"), base_url=server.url)

    # model present in the list
    await validate_builtin_provider(hass, provider, "mimo-v2.5-asr", "key")
    assert server.models_requests[0]["path"].endswith("/v1/models")

    # model missing from the list
    with pytest.raises(WhisperModelNotFound):
        await validate_builtin_provider(hass, provider, "mimo-v2.5-asr", "key")

    # an unparseable body is accepted (fails loudly at runtime instead)
    await validate_builtin_provider(hass, provider, "mimo-v2.5-asr", "key")
    # an empty model list simply means the model is missing
    with pytest.raises(WhisperModelNotFound):
        await validate_builtin_provider(hass, provider, "mimo-v2.5-asr", "key")

    # 404 from the list endpoint is a missing model
    with pytest.raises(WhisperModelNotFound):
        await validate_builtin_provider(hass, provider, "mimo-v2.5-asr", "key")


async def test_validate_builtin_provider_without_key(
    hass: Any, transcription_server_factory: Any
) -> None:
    """No API key means no auth header."""
    from custom_components.openai_whisper_cloud.config_flow import (
        validate_builtin_provider,
    )

    server = await transcription_server_factory(
        [], models_responses=[(200, {"id": "whisper-1"})]
    )
    provider = dataclasses.replace(get_provider("openai"), base_url=server.url)
    await validate_builtin_provider(hass, provider, "whisper-1", "")
    assert server.models_requests[0]["auth"] is None


# ---------------------------------------------------------------------------
# custom provider validation
# ---------------------------------------------------------------------------


async def test_validate_custom_provider_accepts_models_endpoint(
    hass: Any, transcription_server_factory: Any
) -> None:
    """Any non-404 answer from the models endpoint is accepted."""
    server = await transcription_server_factory([], models_responses=[(200, {})])
    await validate_custom_provider(hass, server.url, "key")
    await validate_custom_provider(hass, server.url, "")
    assert len(server.models_requests) == 2


async def test_validate_custom_provider_versioned_base(
    hass: Any, transcription_server_factory: Any
) -> None:
    """A versioned base URL falls back to the unversioned models path."""
    server = await transcription_server_factory([], models_responses=[(200, {})])
    await validate_custom_provider(hass, f"{server.url}/v1", "key")
    assert server.models_requests[0]["path"] == "/v1/models"


async def test_validate_custom_provider_invalid_key(
    hass: Any, transcription_server_factory: Any
) -> None:
    """401 answers are an API key error."""
    server = await transcription_server_factory([], models_responses=[(401, "bad")])
    with pytest.raises(InvalidAPIKey):
        await validate_custom_provider(hass, server.url, "key")


async def test_validate_custom_provider_all_404_accepted(
    hass: Any, transcription_server_factory: Any
) -> None:
    """Servers without a models endpoint are still accepted."""
    server = await transcription_server_factory([], models_responses=[(404, "gone")])
    await validate_custom_provider(hass, f"{server.url}/v1", "key")
    # the versioned candidate was probed; the other one 404s at the router
    # level and does not reach the models handler
    assert server.models_requests[0]["path"] == "/v1/models"


async def test_validate_custom_provider_unreachable(
    hass: Any, transcription_server_factory: Any
) -> None:
    """A dead endpoint is a connection error."""
    server = await transcription_server_factory([])
    url = server.url
    await server.stop()
    with pytest.raises(CannotConnect):
        await validate_custom_provider(hass, url, "key")


# ---------------------------------------------------------------------------
# config flow: user and provider steps
# ---------------------------------------------------------------------------


async def test_user_step_shows_providers(hass: Any) -> None:
    """The first step lists every provider."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    _, validator = get_field(result, CONF_PROVIDER_TYPE)
    values = {option["value"] for option in validator.config["options"]}
    assert values == {
        "openai",
        "groqcloud",
        "mistral",
        "cohere",
        "xiaomimimo",
        "custom",
    }


async def test_user_step_selects_builtin(hass: Any) -> None:
    """Selecting a builtin provider shows its setup form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"provider_type": "openai"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "whisper"
    assert schema_fields(result) == {
        "name",
        "api_key",
        "model",
        "temperature",
        "prompt",
        "streaming_upload",
    }


async def test_user_step_selects_custom(hass: Any) -> None:
    """Selecting the custom provider shows the endpoint form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"provider_type": "custom"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "custom"
    assert CONF_URL in schema_fields(result)


async def test_builtin_schema_hides_unsupported_options(hass: Any) -> None:
    """MiMo and Cohere only offer what their API supports."""
    from custom_components.openai_whisper_cloud.config_flow import _builtin_schema

    assert schema_fields(
        {"data_schema": _builtin_schema(get_provider("cohere"), True)}
    ) == {
        "name",
        "api_key",
        "model",
    }
    assert schema_fields(
        {"data_schema": _builtin_schema(get_provider("xiaomimimo"), True)}
    ) == {"name", "api_key", "model"}


async def test_whisper_step_success(hass: Any) -> None:
    """A valid submission creates the entry with all options."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(),
    ) as mock_validate:
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "openai"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "My Whisper",
                "api_key": "k",
                "model": "whisper-1",
                "temperature": 0.4,
                "prompt": VALID_TEMPLATE,
                "streaming_upload": False,
            },
        )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "My Whisper"
    assert result["data"] == {
        CONF_PROVIDER_TYPE: "openai",
        CONF_NAME: "My Whisper",
        CONF_API_KEY: "k",
    }
    assert result["options"] == {
        CONF_MODEL: "whisper-1",
        CONF_TEMPERATURE: 0.4,
        CONF_PROMPT: VALID_TEMPLATE,
        CONF_STREAMING: False,
    }
    mock_validate.assert_awaited_once()


async def test_whisper_step_defaults_applied(hass: Any) -> None:
    """Missing optional fields fall back to documented defaults."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "groqcloud"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "Groq", "api_key": "k", "model": "whisper-large-v3"},
        )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["options"][CONF_TEMPERATURE] == DEFAULT_TEMPERATURE
    assert result["options"][CONF_PROMPT] == DEFAULT_PROMPT
    assert result["options"][CONF_STREAMING] == DEFAULT_STREAMING


@pytest.mark.parametrize(
    ("error", "field", "error_key"),
    [
        (CannotConnect, "base", "connection_error"),
        (InvalidAPIKey, "api_key", "invalid_api_key"),
        (UnauthorizedError, "base", "unauthorized"),
        (WhisperModelNotFound, "base", "whisper_not_found"),
    ],
)
async def test_whisper_step_provider_errors(
    hass: Any, error: Any, field: str, error_key: str
) -> None:
    """Provider validation failures show the right form errors."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(side_effect=error),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "openai"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "W", "api_key": "k", "model": "whisper-1"},
        )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "whisper"
    assert result["errors"] == {field: error_key}


async def test_whisper_step_invalid_prompt_no_provider_call(hass: Any) -> None:
    """An invalid prompt fails fast without contacting the provider."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(),
    ) as mock_validate:
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "openai"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "W",
                "api_key": "k",
                "model": "whisper-1",
                "prompt": "{{ broken",
            },
        )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {CONF_PROMPT: "invalid_prompt_template"}
    assert "TemplateSyntaxError" in result["description_placeholders"]["error_detail"]
    marker, _ = get_field(result, CONF_PROMPT)
    assert marker.description["suggested_value"] == "{{ broken"
    mock_validate.assert_not_awaited()


async def test_custom_step_success(hass: Any) -> None:
    """A valid custom endpoint creates the entry."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_custom_provider",
        new=AsyncMock(),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "custom"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Local",
                "url": "http://127.0.0.1:8000/asr/",
                "model": "whisper-1",
                "file_field": "audio_file",
                "prompt": "hello",
            },
        )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_BASE_URL] == "http://127.0.0.1:8000/asr"
    assert result["data"][CONF_FILE_FIELD] == "audio_file"
    assert result["options"] == {
        CONF_MODEL: "whisper-1",
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_PROMPT: "hello",
        CONF_STREAMING: DEFAULT_STREAMING,
    }


async def test_custom_step_errors(hass: Any) -> None:
    """URL and connection errors show the right form errors."""
    # invalid URL
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"provider_type": "custom"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"name": "L", "url": "not a url", "model": "whisper-1"},
    )
    assert result["errors"] == {"url": "invalid_url"}

    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_custom_provider",
        new=AsyncMock(side_effect=CannotConnect),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "L", "url": "http://127.0.0.1:8000", "model": "whisper-1"},
        )
    assert result["errors"] == {"base": "connection_error"}

    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_custom_provider",
        new=AsyncMock(side_effect=InvalidAPIKey),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "L", "url": "http://127.0.0.1:8000", "model": "whisper-1"},
        )
    assert result["errors"] == {"api_key": "invalid_api_key"}


async def test_custom_step_invalid_prompt(hass: Any) -> None:
    """An invalid prompt is caught before the endpoint is contacted."""
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_custom_provider",
        new=AsyncMock(),
    ) as mock_validate:
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"provider_type": "custom"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "L",
                "url": "http://127.0.0.1:8000",
                "model": "m",
                "prompt": "{% bad",
            },
        )
    assert result["errors"] == {CONF_PROMPT: "invalid_prompt_template"}
    assert "error_detail" in result["description_placeholders"]
    marker, _ = get_field(result, CONF_PROMPT)
    assert marker.description["suggested_value"] == "{% bad"
    mock_validate.assert_not_awaited()


# ---------------------------------------------------------------------------
# reconfigure flow
# ---------------------------------------------------------------------------


def builtin_entry() -> MockConfigEntry:
    """A configured OpenAI entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        title="Old",
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "Old", CONF_API_KEY: "k"},
        options={
            CONF_MODEL: "whisper-1",
            CONF_TEMPERATURE: 0.0,
            CONF_PROMPT: "old",
            CONF_STREAMING: True,
        },
    )


def custom_entry() -> MockConfigEntry:
    """A configured custom entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        title="Local",
        data={
            CONF_PROVIDER_TYPE: "custom",
            CONF_NAME: "Local",
            CONF_BASE_URL: "http://127.0.0.1:8000/asr",
            CONF_API_KEY: "",
            CONF_FILE_FIELD: "file",
        },
        options={CONF_MODEL: "m", CONF_TEMPERATURE: 0.0, CONF_PROMPT: ""},
    )


async def start_reconfigure(hass: Any, entry: MockConfigEntry) -> dict[str, Any]:
    """Open the reconfigure flow for an entry."""
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "reconfigure", "entry_id": entry.entry_id},
        data=None,
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    return result


async def test_reconfigure_builtin_success(hass: Any) -> None:
    """Reconfigure updates data and options."""
    entry = builtin_entry()
    result = await start_reconfigure(hass, entry)
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Renamed",
                "model": "whisper-1",
                "temperature": 0.5,
                "prompt": "new prompt",
                "streaming_upload": False,
            },
        )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_NAME] == "Renamed"
    assert entry.options == {
        CONF_MODEL: "whisper-1",
        CONF_TEMPERATURE: 0.5,
        CONF_PROMPT: "new prompt",
        CONF_STREAMING: False,
    }


async def test_reconfigure_custom_success(hass: Any) -> None:
    """Reconfigure of a custom entry keeps and updates its data."""
    entry = custom_entry()
    result = await start_reconfigure(hass, entry)
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_custom_provider",
        new=AsyncMock(),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Local",
                "url": "http://127.0.0.1:9000/asr",
                "model": "new-model",
                "file_field": "audio_file",
            },
        )
    assert result["type"] == FlowResultType.ABORT
    assert entry.data[CONF_BASE_URL] == "http://127.0.0.1:9000/asr"
    assert entry.data[CONF_FILE_FIELD] == "audio_file"
    assert entry.options[CONF_MODEL] == "new-model"


async def test_reconfigure_errors(hass: Any) -> None:
    """Reconfigure shows form errors and keeps the entry untouched."""
    entry = builtin_entry()
    result = await start_reconfigure(hass, entry)
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(side_effect=CannotConnect),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "X", "model": "whisper-1"},
        )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "connection_error"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"name": "X", "model": "whisper-1", "prompt": "{% bad"},
    )
    assert result["errors"] == {CONF_PROMPT: "invalid_prompt_template"}
    assert "error_detail" in result["description_placeholders"]
    assert entry.options[CONF_PROMPT] == "old"


async def test_reconfigure_suggested_values(hass: Any) -> None:
    """The form is prefilled with the current entry values."""
    entry = builtin_entry()
    result = await start_reconfigure(hass, entry)
    marker, _ = get_field(result, CONF_MODEL)
    assert marker.description["suggested_value"] == "whisper-1"
    marker, _ = get_field(result, CONF_PROMPT)
    assert marker.description["suggested_value"] == "old"
    marker, _ = get_field(result, CONF_STREAMING)
    assert marker.description["suggested_value"] is True


# ---------------------------------------------------------------------------
# options flow
# ---------------------------------------------------------------------------


async def test_options_flow_updates_options(hass: Any) -> None:
    """The options flow persists model, temperature, prompt, streaming."""
    entry = builtin_entry()
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    assert schema_fields(result) == {
        "model",
        "temperature",
        "prompt",
        "streaming_upload",
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "model": "gpt-4o-mini-transcribe",
            "temperature": 0.3,
            "prompt": VALID_TEMPLATE,
            "streaming_upload": False,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options == {
        CONF_MODEL: "gpt-4o-mini-transcribe",
        CONF_TEMPERATURE: 0.3,
        CONF_PROMPT: VALID_TEMPLATE,
        CONF_STREAMING: False,
    }


async def test_options_flow_invalid_prompt_keeps_input(hass: Any) -> None:
    """An invalid prompt re-shows the form with the typed input intact."""
    entry = builtin_entry()
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"model": "whisper-1", "temperature": 0, "prompt": "{{ oops"},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {CONF_PROMPT: "invalid_prompt_template"}
    assert "TemplateSyntaxError" in result["description_placeholders"]["error_detail"]
    marker, _ = get_field(result, CONF_PROMPT)
    assert marker.description["suggested_value"] == "{{ oops"


async def test_options_flow_first_render_has_defaults(hass: Any) -> None:
    """The initial form has no suggested values, only defaults."""
    entry = builtin_entry()
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    marker, validator = get_field(result, CONF_PROMPT)
    assert marker.default() == "old"
    assert not marker.description or "suggested_value" not in marker.description
    assert validator.config["multiline"] is True


async def test_options_flow_custom_provider(hass: Any) -> None:
    """Custom providers offer a free-text model and all options."""
    entry = custom_entry()
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert schema_fields(result) == {
        "model",
        "temperature",
        "prompt",
        "streaming_upload",
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "model": "other-model",
            "temperature": 0.1,
            "prompt": "",
            "streaming_upload": True,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_MODEL] == "other-model"


async def test_options_flow_restricted_provider(hass: Any) -> None:
    """MiMo entries only offer the model."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={CONF_PROVIDER_TYPE: "xiaomimimo", CONF_NAME: "MiMo", CONF_API_KEY: "k"},
        options={CONF_MODEL: "mimo-v2.5-asr"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert schema_fields(result) == {"model"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"model": "mimo-v2.5-asr"}
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options == {CONF_MODEL: "mimo-v2.5-asr"}


async def test_options_flow_cohere_provider(hass: Any) -> None:
    """Cohere entries only offer the model."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={CONF_PROVIDER_TYPE: "cohere", CONF_NAME: "Cohere", CONF_API_KEY: "k"},
        options={CONF_MODEL: "cohere-transcribe-03-2026"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert schema_fields(result) == {"model"}


async def test_reconfigure_custom_invalid_url(hass: Any) -> None:
    """An invalid URL during reconfigure shows the url error."""
    entry = custom_entry()
    result = await start_reconfigure(hass, entry)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"name": "L", "url": "not a url", "model": "m"},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"url": "invalid_url"}


@pytest.mark.parametrize(
    ("error", "field", "error_key"),
    [
        (InvalidAPIKey, "api_key", "invalid_api_key"),
        (UnauthorizedError, "base", "unauthorized"),
        (WhisperModelNotFound, "base", "whisper_not_found"),
    ],
)
async def test_reconfigure_provider_errors(
    hass: Any, error: Any, field: str, error_key: str
) -> None:
    """Provider validation failures during reconfigure show form errors."""
    entry = builtin_entry()
    result = await start_reconfigure(hass, entry)
    with patch(
        "custom_components.openai_whisper_cloud.config_flow.validate_builtin_provider",
        new=AsyncMock(side_effect=error),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"name": "X", "model": "whisper-1"},
        )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {field: error_key}


async def test_options_flow_model_no_longer_available(hass: Any) -> None:
    """A stored model missing from the provider falls back to its default."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={CONF_PROVIDER_TYPE: "openai", CONF_NAME: "N", CONF_API_KEY: "k"},
        options={CONF_MODEL: "whisper-deprecated"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    marker, validator = get_field(result, CONF_MODEL)
    assert marker.default() == get_provider("openai").default_model
