"""Tests for the STT entity."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.openai_whisper_cloud.const import (
    CONF_BASE_URL,
    CONF_FILE_FIELD,
    CONF_PROVIDER_TYPE,
    CONF_STREAMING,
    CONF_TEMPERATURE,
    DOMAIN,
)
from custom_components.openai_whisper_cloud.stt import (
    ASSIST_ENTITIES,
    ASSIST_ENTITY_NAMES,
    OpenAIWhisperEntity,
    assist_prompt_variables,
    resolve_language,
    resolve_prompt,
)
from custom_components.openai_whisper_cloud.whisper_provider import get_provider

from .conftest import TEST_TRANSCRIPTION

# These tests talk to scripted local HTTP servers; the plugin restricts
# real sockets to 127.0.0.1.
pytestmark = pytest.mark.usefixtures("socket_enabled")


PCM_CHUNK = b"\x01\x02" * 16000


def make_metadata(language: str | None = "en") -> Any:
    """Build STT metadata like the assist pipeline does."""
    from homeassistant.components.stt import (
        AudioBitRates,
        AudioChannels,
        AudioCodecs,
        AudioFormats,
        AudioSampleRates,
        SpeechMetadata,
    )

    return SpeechMetadata(
        language=language,
        format=AudioFormats.WAV,
        codec=AudioCodecs.PCM,
        bit_rate=AudioBitRates.BITRATE_16,
        sample_rate=AudioSampleRates.SAMPLERATE_16000,
        channel=AudioChannels.CHANNEL_MONO,
    )


async def fast_stream(chunks: int = 3) -> Any:
    """Produce audio immediately."""
    for _ in range(chunks):
        yield PCM_CHUNK


# ---------------------------------------------------------------------------
# language resolution
# ---------------------------------------------------------------------------


def test_resolve_language_passthrough() -> None:
    """Supported languages pass through unchanged."""
    supported = get_provider("openai").models[0].languages
    assert resolve_language("en", supported) == "en"
    assert resolve_language("EN", supported) == "en"


def test_resolve_language_regional_folds_to_base() -> None:
    """Regional variants fold onto the supported base language."""
    supported = get_provider("openai").models[0].languages
    assert resolve_language("pt-BR", supported) == "pt"
    assert resolve_language("en-GB", supported) == "en"


def test_resolve_language_unsupported_omitted() -> None:
    """Unsupported languages map to None (provider auto-detection)."""
    supported = get_provider("openai").models[0].languages
    assert resolve_language(None, supported) is None
    assert resolve_language("", supported) is None
    assert resolve_language("xx-YY", supported) is None


def test_resolve_language_mimo_chinese_variants() -> None:
    """MiMo declares Chinese variants for the pipeline metadata check."""
    supported = get_provider("xiaomimimo").models[0].languages
    assert resolve_language("zh-hans", supported) == "zh-hans"
    assert resolve_language("zh-cn", supported) == "zh-cn"
    # unknown variants fold to zh when listed, or are dropped otherwise
    assert resolve_language("zh-SG", supported) == "zh"


# ---------------------------------------------------------------------------
# prompt resolution
# ---------------------------------------------------------------------------


async def test_resolve_prompt_passthrough(hass: Any) -> None:
    """Non-template prompts are returned unchanged."""
    assert resolve_prompt(hass, "") == ""
    assert resolve_prompt(hass, "living room light, kitchen") == (
        "living room light, kitchen"
    )
    # a lone brace is not template syntax
    assert resolve_prompt(hass, "a { b") == "a { b"


async def test_resolve_prompt_template(hass: Any) -> None:
    """Prompt templates are rendered against the current state."""
    hass.states.async_set("sensor.entity_names", "Living Room Light, Kitchen Door")
    assert (
        resolve_prompt(hass, "{{ states('sensor.entity_names') }}, Garage")
        == "Living Room Light, Kitchen Door, Garage"
    )


async def test_resolve_prompt_broken_template(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken template transcribes without a prompt."""
    assert resolve_prompt(hass, "{{ broken") == ""
    assert "failed to render" in caplog.text


async def setup_assist_exposure(hass: Any) -> tuple[str, str]:
    """Expose one light with an alias and hide one sensor from Assist."""
    assert await async_setup_component(hass, "homeassistant", {})
    registry = er.async_get(hass)

    light = registry.async_get_or_create("light", "test", "living-room")
    registry.async_update_entity(light.entity_id, aliases=["Lounge Lamp"])
    hass.states.async_set(light.entity_id, "on", {"friendly_name": "Living Room Light"})
    async_expose_entity(hass, "conversation", light.entity_id, True)

    sensor = registry.async_get_or_create("sensor", "test", "secret")
    hass.states.async_set(sensor.entity_id, "42", {"friendly_name": "Secret Sensor"})
    async_expose_entity(hass, "conversation", sensor.entity_id, False)
    return light.entity_id, sensor.entity_id


async def test_assist_prompt_variables(hass: Any) -> None:
    """Exposed entity names and registry aliases are collected for Assist."""
    light_entity_id, _ = await setup_assist_exposure(hass)

    variables = assist_prompt_variables(hass)

    assert variables[ASSIST_ENTITY_NAMES] == "Living Room Light, Lounge Lamp"
    assert variables[ASSIST_ENTITIES] == [
        {
            "entity_id": light_entity_id,
            "domain": "light",
            "names": ["Living Room Light", "Lounge Lamp"],
        }
    ]


async def test_resolve_prompt_assist_variables(hass: Any) -> None:
    """The Assist helpers are available inside prompt templates."""
    await setup_assist_exposure(hass)

    assert (
        resolve_prompt(hass, "{{ assist_entity_names }}")
        == "Living Room Light, Lounge Lamp"
    )
    assert (
        resolve_prompt(
            hass,
            "{% for entity in assist_entities %}"
            "{{ entity.names | join(' / ') }}"
            "{% if not loop.last %} | {% endif %}"
            "{% endfor %}",
        )
        == "Living Room Light / Lounge Lamp"
    )


# ---------------------------------------------------------------------------
# entity
# ---------------------------------------------------------------------------


def entity(hass: Any, url: str | None = None, **kwargs: Any) -> OpenAIWhisperEntity:
    """Build a transcription entity (local server URL, or OpenAI by default)."""
    defaults: dict[str, Any] = {
        "api_key": "k",
        "temperature": 0.0,
        "prompt": "",
        "model_name": None,
        "streaming": True,
        "name": "Whisper",
        "unique_id": "test-uid",
    }
    defaults.update(kwargs)
    if url:
        from custom_components.openai_whisper_cloud.whisper_provider import (
            custom_provider,
        )

        provider = custom_provider(url, "whisper-1")
    else:
        provider = get_provider("openai")
    entity = OpenAIWhisperEntity(provider, **defaults)
    entity.hass = hass
    return entity


async def test_entity_supported_properties(hass: Any) -> None:
    """The entity advertises what it accepts."""
    test_entity = entity(hass)
    assert test_entity.supported_languages == get_provider("openai").models[0].languages
    from homeassistant.components.stt import (
        AudioBitRates,
        AudioChannels,
        AudioCodecs,
        AudioFormats,
        AudioSampleRates,
    )

    assert test_entity.supported_formats == [AudioFormats.WAV]
    assert test_entity.supported_codecs == [AudioCodecs.PCM]
    assert test_entity.supported_bit_rates == [
        AudioBitRates.BITRATE_8,
        AudioBitRates.BITRATE_16,
        AudioBitRates.BITRATE_24,
        AudioBitRates.BITRATE_32,
    ]
    assert test_entity.supported_sample_rates == [
        AudioSampleRates.SAMPLERATE_8000,
        AudioSampleRates.SAMPLERATE_16000,
        AudioSampleRates.SAMPLERATE_44100,
        AudioSampleRates.SAMPLERATE_48000,
    ]
    assert test_entity.supported_channels == [
        AudioChannels.CHANNEL_MONO,
        AudioChannels.CHANNEL_STEREO,
    ]


async def test_entity_attributes(hass: Any) -> None:
    """The entity exposes activity and configuration attributes."""
    test_entity = entity(hass, streaming=False, name="My Whisper")
    attrs = test_entity.extra_state_attributes
    assert attrs == {
        "processing": False,
        "provider": "OpenAI",
        "model": "whisper-1",
        "streaming_upload": False,
    }


async def test_entity_device_info(hass: Any) -> None:
    """The device is described by provider and model."""
    test_entity = entity(hass, name="My Whisper")
    assert test_entity.unique_id == "test-uid"
    assert test_entity.name == "My Whisper"
    device_info = test_entity.device_info
    assert device_info["identifiers"] == {(DOMAIN, "test-uid")}
    assert device_info["manufacturer"] == "OpenAI"
    assert device_info["model"] == "whisper-1"


async def test_entity_transcribes(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """The entity turns audio into a successful speech result."""
    from homeassistant.components.stt import SpeechResultState

    hass.states.async_set("sensor.rooms", "kitchen, garage")
    server = await transcription_server_factory([(200, None)])
    test_entity = entity(hass, url=server.url, prompt="{{ states('sensor.rooms') }}")
    result = await test_entity.async_process_audio_stream(
        make_metadata(), fast_stream()
    )
    assert result.result is SpeechResultState.SUCCESS
    assert result.text == TEST_TRANSCRIPTION
    # the prompt template was rendered and sent with the request
    assert server.requests[0]["fields"]["prompt"] == "kitchen, garage"


async def test_entity_sends_assist_prompt(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Assist entity names are sent to the provider like any other prompt."""
    from homeassistant.components.stt import SpeechResultState

    await setup_assist_exposure(hass)
    server = await transcription_server_factory([(200, None)])
    test_entity = entity(hass, url=server.url, prompt="{{ assist_entity_names }}")
    result = await test_entity.async_process_audio_stream(
        make_metadata(), fast_stream()
    )
    assert result.result is SpeechResultState.SUCCESS
    assert server.requests[0]["fields"]["prompt"] == ("Living Room Light, Lounge Lamp")


async def test_entity_state_bookkeeping(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """The state timestamp and processing flag track activity."""
    from homeassistant.components.stt import SpeechResultState

    server = await transcription_server_factory([(200, None)])
    server.delay = 0.3
    test_entity = entity(hass, url=server.url)

    # not yet added to hass: state writes are skipped, tracking still works
    assert test_entity.state is None
    assert test_entity._async_write_activity_state() is None

    task = asyncio.create_task(
        test_entity.async_process_audio_stream(make_metadata(), fast_stream())
    )
    await asyncio.sleep(0.05)
    assert test_entity.extra_state_attributes["processing"] is True
    timestamp = test_entity.state
    assert timestamp is not None and "T" in timestamp
    result = await task
    assert result.result is SpeechResultState.SUCCESS
    assert test_entity.extra_state_attributes["processing"] is False
    assert test_entity.state == timestamp
    assert test_entity._active == 0
    server.delay = 0.0

    # a failed transcription still leaves the entity consistent
    server.responses = [(400, "nope")]
    result = await test_entity.async_process_audio_stream(
        make_metadata(), fast_stream()
    )
    assert result.result is SpeechResultState.ERROR
    assert test_entity.extra_state_attributes["processing"] is False
    assert test_entity._active == 0


async def test_entity_client_is_cached(
    hass: Any,
) -> None:
    """The transcription client is created once and reused."""
    test_entity = entity(hass)
    client = test_entity._get_client()
    assert test_entity._get_client() is client


async def test_setup_entry_creates_entity(hass: Any) -> None:
    """Setting up the config entry registers the STT entity."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={
            CONF_PROVIDER_TYPE: "openai",
            CONF_NAME: "Cloud Whisper",
            CONF_API_KEY: "k",
        },
        options={
            CONF_MODEL: "whisper-1",
            CONF_TEMPERATURE: 0.2,
            "prompt": "hello",
            CONF_STREAMING: False,
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("stt.cloud_whisper")
    assert state is not None
    assert state.attributes["processing"] is False
    assert state.attributes["model"] == "whisper-1"
    assert state.attributes["streaming_upload"] is False
    assert state.state == "unknown"


async def test_setup_entry_custom_provider(hass: Any) -> None:
    """Custom entries build their provider from the URL and file field."""
    from custom_components.openai_whisper_cloud.whisper_provider import (
        resolve_transcription_url,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={
            CONF_PROVIDER_TYPE: "custom",
            CONF_NAME: "Local Whisper",
            CONF_BASE_URL: "http://127.0.0.1:8000/asr",
            CONF_API_KEY: "",
            CONF_FILE_FIELD: "audio_file",
        },
        options={CONF_MODEL: "distil-medium.en"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("stt.local_whisper")
    assert state is not None
    assert state.attributes["provider"] == "Custom"
    assert state.attributes["model"] == "distil-medium.en"
    assert resolve_transcription_url("http://127.0.0.1:8000/asr") == (
        "http://127.0.0.1:8000/asr"
    )
    assert get_provider("custom").multipart_file_field == "file"


async def test_registered_entity_writes_state(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """An entity registered with the stt component updates its state."""
    from homeassistant.components.stt import DATA_COMPONENT
    from homeassistant.components.stt import SpeechResultState

    server = await transcription_server_factory([(200, None)])
    server.delay = 0.3
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={
            CONF_PROVIDER_TYPE: "custom",
            CONF_NAME: "Stateful Whisper",
            CONF_BASE_URL: server.url,
            CONF_API_KEY: "",
        },
        options={CONF_MODEL: "whisper-1"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity = next(
        e
        for e in hass.data[DATA_COMPONENT].entities
        if isinstance(e, OpenAIWhisperEntity)
    )
    assert entity.entity_id == "stt.stateful_whisper"

    task = asyncio.create_task(
        entity.async_process_audio_stream(make_metadata(), fast_stream())
    )
    await asyncio.sleep(0.05)
    state = hass.states.get(entity.entity_id)
    assert state.attributes["processing"] is True
    assert "T" in state.state  # last-processed timestamp

    result = await task
    assert result.result is SpeechResultState.SUCCESS
    state = hass.states.get(entity.entity_id)
    assert state.attributes["processing"] is False
    assert state.state != "unknown"
