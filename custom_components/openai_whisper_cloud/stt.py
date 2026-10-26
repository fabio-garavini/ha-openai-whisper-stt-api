"""OpenAI Whisper API speech-to-text entity."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import Any, Final

from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.template import Template, TemplateError, is_template_string
from homeassistant.util import dt as dt_util

from .client import WhisperTranscriptionClient
from .const import (
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
from .whisper_provider import (
    WhisperModel,
    WhisperProvider,
    custom_provider,
    get_provider,
)

_LOGGER = logging.getLogger(__name__)

# The assistant whose exposure settings decide which entities Assist can
# actually talk about.
ASSIST_ASSISTANT: Final = "conversation"
# Template variables made available to prompt templates.
ASSIST_ENTITY_NAMES: Final = "assist_entity_names"
ASSIST_ENTITIES: Final = "assist_entities"


def resolve_language(language: str | None, supported: list[str]) -> str | None:
    """Map a requested language onto one supported by the model.

    Regional variants (e.g. ``pt-br``) are folded onto their base language
    (``pt``) when the base language is supported. This is needed because
    Whisper-style APIs only accept base language codes, while Home Assistant
    may request regional variants for its intent system.

    A language the model does not support at all maps to ``None`` so the
    provider auto-detects the spoken language instead of failing the
    request with an unsupported language code.
    """
    if not language:
        return None
    lang = language.lower()
    if lang in supported:
        return lang
    base = lang.split("-")[0]
    if base in supported:
        return base
    return None


@callback
def assist_prompt_variables(hass: HomeAssistant) -> dict[str, Any]:
    """Return template variables describing the entities Assist can control.

    ``assist_entity_names`` is a ready-to-use, comma separated string with
    every name and alias Assist matches for entities exposed to it, so a
    prompt can simply be ``{{ assist_entity_names }}``.

    ``assist_entities`` exposes the same information as a list of
    ``{"entity_id", "domain", "names"}`` dictionaries for templates that
    need to filter or format the list themselves.

    Home Assistant only exposes entities to Assist; device names are part
    of the computed entity names for devices that use the entity name, and
    the entity registry aliases are included as well.
    """
    if DATA_EXPOSED_ENTITIES not in hass.data:
        # The homeassistant component is not set up yet (e.g. in unit tests).
        return {ASSIST_ENTITY_NAMES: "", ASSIST_ENTITIES: []}

    entity_registry = er.async_get(hass)
    # Resolves the computed device+entity name on current Home Assistant
    # versions. Older versions only have the plain string aliases.
    get_entity_aliases = getattr(intent, "async_get_entity_aliases", None)
    entities: list[dict[str, Any]] = []
    prompt_names: list[str] = []
    seen_prompt_names: set[str] = set()

    for state in hass.states.async_all():
        if not async_should_expose(hass, ASSIST_ASSISTANT, state.entity_id):
            continue

        entity_entry = entity_registry.async_get(state.entity_id)
        names: list[str] = []
        seen_entity_names: set[str] = set()
        # Include the state name as well as the registry aliases: aliases
        # contain the computed device+entity name on current Home Assistant
        # versions, while the state name covers older entries and custom
        # names.
        if get_entity_aliases is not None:
            aliases = get_entity_aliases(hass, entity_entry, state=state)
        elif entity_entry is not None:
            aliases = [
                alias for alias in entity_entry.aliases if isinstance(alias, str)
            ]
        else:
            aliases = []
        for name in (state.name, *aliases):
            name = (name or "").strip()
            key = name.casefold()
            if not name or key in seen_entity_names:
                continue
            seen_entity_names.add(key)
            names.append(name)

        if not names:
            continue

        entities.append(
            {
                "entity_id": state.entity_id,
                "domain": state.domain,
                "names": names,
            }
        )
        for name in names:
            key = name.casefold()
            if key not in seen_prompt_names:
                seen_prompt_names.add(key)
                prompt_names.append(name)

    entities.sort(key=lambda entity: entity["entity_id"])
    prompt_names.sort(key=str.casefold)
    _LOGGER.debug(
        "Assist exposes %d entities with %d distinct names/aliases",
        len(entities),
        len(prompt_names),
    )
    return {
        ASSIST_ENTITY_NAMES: ", ".join(prompt_names),
        ASSIST_ENTITIES: entities,
    }


@callback
def resolve_prompt(hass: HomeAssistant, prompt: str) -> str:
    """Return the prompt to send with a transcription request.

    The prompt may contain Home Assistant templates (e.g. built from sensors
    that list entity names), so it can stay in sync with the home without
    manual edits. Templates are rendered fresh for every transcription,
    which means the prompt always reflects the current state and entity
    names are picked up as soon as they are added or renamed.

    Prompt templates get the ``assist_entity_names`` and ``assist_entities``
    variables, which describe the entities and aliases exposed to Assist.

    A prompt without template syntax is returned unchanged. A template
    that fails to render is logged and skipped instead of failing the
    whole transcription.
    """
    if not prompt or not is_template_string(prompt):
        return prompt

    variables: dict[str, Any] | None = None
    if ASSIST_ENTITY_NAMES in prompt or ASSIST_ENTITIES in prompt:
        variables = assist_prompt_variables(hass)
    try:
        rendered = Template(prompt, hass).async_render(
            variables=variables, parse_result=False
        )
    except TemplateError as err:
        _LOGGER.warning(
            "Transcribing without a prompt because its template failed to render: %s",
            err,
        )
        return ""
    _LOGGER.debug("Rendered prompt template to %d characters", len(rendered))
    return rendered


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Whisper speech-to-text platform via config entry."""
    _LOGGER.debug("Setting up STT entity for entry %s", config_entry.entry_id)

    provider_type = config_entry.data[CONF_PROVIDER_TYPE]
    if provider_type == "custom":
        provider = custom_provider(
            config_entry.data.get(CONF_BASE_URL, ""),
            config_entry.options.get(CONF_MODEL, "whisper-1"),
            file_field=config_entry.data.get(CONF_FILE_FIELD, "file"),
        )
    else:
        provider = get_provider(provider_type)

    async_add_entities(
        [
            OpenAIWhisperEntity(
                provider=provider,
                api_key=config_entry.data.get(CONF_API_KEY, ""),
                temperature=float(
                    config_entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
                ),
                prompt=config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                model_name=config_entry.options.get(CONF_MODEL),
                streaming=bool(
                    config_entry.options.get(CONF_STREAMING, DEFAULT_STREAMING)
                ),
                name=config_entry.data[CONF_NAME],
                unique_id=config_entry.entry_id,
            )
        ]
    )


class OpenAIWhisperEntity(SpeechToTextEntity):
    """OpenAI Whisper API provider entity."""

    def __init__(
        self,
        provider: WhisperProvider,
        *,
        api_key: str,
        temperature: float,
        prompt: str,
        model_name: str | None,
        streaming: bool,
        name: str,
        unique_id: str,
    ) -> None:
        """Init the STT entity."""
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.prompt = prompt
        self.streaming = streaming
        self.model: WhisperModel = provider.get_model(model_name)
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, unique_id)},
            name=name,
            manufacturer=provider.name,
            model=self.model.name,
        )
        self._client: WhisperTranscriptionClient | None = None
        self._active = 0

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return self.model.languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bit rates."""
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return activity and configuration details."""
        return {
            "processing": self._active > 0,
            "provider": self.provider.name,
            "model": self.model.name,
            "streaming_upload": self.streaming,
        }

    def _get_client(self) -> WhisperTranscriptionClient:
        """Return the transcription client, creating it on first use."""
        if self._client is None:
            self._client = WhisperTranscriptionClient(
                self.hass,
                self.provider,
                api_key=self.api_key,
                model_name=self.model.name,
                temperature=self.temperature,
                streaming=self.streaming,
            )
        return self._client

    @callback
    def _async_write_activity_state(self) -> None:
        """Write the state once the entity has been added to hass."""
        if self.entity_id:
            self.async_write_ha_state()

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to text.

        The framework defines the STT entity state as the time the last
        transcription started, but only its internal wrapper (used by the
        HTTP API) maintains it: the assist pipeline calls this public
        method directly. Do the same bookkeeping here so the state is
        maintained whichever entry point is used, and mirror live activity
        in the ``processing`` attribute.
        """
        self._SpeechToTextEntity__last_processed = dt_util.utcnow().isoformat()
        self._active += 1
        self._async_write_activity_state()
        _LOGGER.debug("Processing audio stream: %s", metadata)
        try:
            language = resolve_language(metadata.language, self.model.languages)
            prompt = resolve_prompt(self.hass, self.prompt)
            text = await self._get_client().transcribe(
                language=language,
                prompt=prompt,
                channels=int(metadata.channel),
                sample_rate=int(metadata.sample_rate),
                stream=stream,
            )
        finally:
            self._active -= 1
            self._async_write_activity_state()
        if text is None:
            return SpeechResult("", SpeechResultState.ERROR)
        _LOGGER.debug("Transcription: %s", text)
        return SpeechResult(text, SpeechResultState.SUCCESS)
