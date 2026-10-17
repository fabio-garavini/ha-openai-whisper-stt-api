"""Tests for the provider definitions."""

from homeassistant.const import CONF_API_KEY

from custom_components.openai_whisper_cloud.const import (
    DEFAULT_STREAMING,
    MAX_AUDIO_SIZE_BYTES,
    MIMO_MAX_AUDIO_BYTES,
)
from custom_components.openai_whisper_cloud.whisper_provider import (
    ApiStyle,
    LEGACY_SOURCE_KEYS,
    MIMO_ASR_LANGUAGES,
    PROVIDERS,
    WhisperProvider,
    custom_provider,
    get_provider,
    resolve_transcription_url,
)


def test_resolve_transcription_url_bare_host() -> None:
    """A bare host gets the standard endpoint appended."""
    assert (
        resolve_transcription_url("https://host")
        == "https://host/v1/audio/transcriptions"
    )


def test_resolve_transcription_url_versioned() -> None:
    """A versioned base only gets the audio path appended."""
    assert (
        resolve_transcription_url("https://host/v1")
        == "https://host/v1/audio/transcriptions"
    )


def test_resolve_transcription_url_full_path() -> None:
    """A URL that already has a path is used as-is."""
    assert (
        resolve_transcription_url("https://host/any/prefix/v1/audio/transcriptions")
        == "https://host/any/prefix/v1/audio/transcriptions"
    )
    assert resolve_transcription_url("http://host:8000/asr") == "http://host:8000/asr"


def test_resolve_transcription_url_strips_whitespace_and_slashes() -> None:
    """Whitespace and trailing slashes are normalized away."""
    assert (
        resolve_transcription_url("  https://host/v1/  ")
        == "https://host/v1/audio/transcriptions"
    )


def test_custom_provider_defaults() -> None:
    """The custom provider keeps URL, model and multipart field."""
    provider = custom_provider("https://example.com", "my-model")
    assert provider.key == "custom"
    assert provider.custom is True
    assert provider.name == "Custom"
    assert provider.default_model == "my-model"
    assert provider.multipart_file_field == "file"
    assert provider.transcription_url == "https://example.com/v1/audio/transcriptions"
    assert provider.api_style == ApiStyle.TRANSCRIPTIONS


def test_custom_provider_file_field() -> None:
    """An explicit multipart field is used, empty falls back to file."""
    assert (
        custom_provider("https://h", "m", file_field="audio_file").multipart_file_field
        == "audio_file"
    )
    assert (
        custom_provider("https://h", "m", file_field="").multipart_file_field == "file"
    )


def test_get_provider_known_and_unknown() -> None:
    """Known keys resolve; unknown keys fall back to OpenAI."""
    assert get_provider("openai").name == "OpenAI"
    assert get_provider("groqcloud").name == "GroqCloud"
    assert get_provider("xiaomimimo").name == "Xiaomi MiMo"
    assert get_provider("does-not-exist").key == "openai"


def test_get_model_exact_match() -> None:
    """The requested model is returned when it exists."""
    provider = get_provider("openai")
    assert provider.get_model("whisper-1").name == "whisper-1"


def test_get_model_fallbacks() -> None:
    """Unknown models fall back to the first model, then to whisper-1."""
    provider = get_provider("groqcloud")
    assert provider.get_model("not-a-model").name == "whisper-large-v3"
    empty = WhisperProvider(key="x", name="X")
    assert empty.get_model(None).name == "whisper-1"
    # no known models: the requested name is kept (custom free-text models)
    assert empty.get_model("anything").name == "anything"


def test_provider_registry_sanity() -> None:
    """Every provider is uniquely keyed and fully configured."""
    assert list(PROVIDERS) == [
        "openai",
        "groqcloud",
        "mistral",
        "cohere",
        "xiaomimimo",
        "custom",
    ]
    for key, provider in PROVIDERS.items():
        assert provider.key == key
        if not provider.custom:
            assert provider.base_url.startswith("https://")
            assert provider.models
            assert provider.default_model in [m.name for m in provider.models]
            assert provider.transcription_url == (
                f"{provider.base_url}{provider.transcription_path}"
            )
    # the legacy migration mapping must never change order
    assert LEGACY_SOURCE_KEYS == ("openai", "groqcloud", "mistral", "cohere")


def test_openai_provider_models() -> None:
    """The OpenAI provider exposes the expected models and defaults."""
    provider = get_provider("openai")
    assert [m.name for m in provider.models] == [
        "whisper-1",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
    ]
    assert provider.default_model == "gpt-4o-mini-transcribe"
    assert provider.supports_streaming is True
    assert provider.max_audio_bytes == MAX_AUDIO_SIZE_BYTES


def test_cohere_provider_capabilities() -> None:
    """Cohere uses the v2 API and supports none of the extra options."""
    provider = get_provider("cohere")
    assert provider.transcription_path == "/v2/audio/transcriptions"
    assert provider.supports_temperature is False
    assert provider.supports_prompt is False
    assert provider.supports_response_format is False
    assert provider.supports_streaming is False


def test_mimo_provider_capabilities() -> None:
    """MiMo serves ASR through chat completions with a tighter size cap."""
    provider = get_provider("xiaomimimo")
    model = provider.get_model(None)
    assert model.name == "mimo-v2.5-asr"
    assert model.languages == MIMO_ASR_LANGUAGES
    assert (
        provider.transcription_url == "https://api.xiaomimimo.com/v1/chat/completions"
    )
    assert provider.api_style == ApiStyle.CHAT_COMPLETIONS
    assert provider.list_models_only is True
    assert provider.supports_temperature is False
    assert provider.supports_prompt is False
    assert provider.supports_response_format is False
    assert provider.supports_streaming is False
    # 10 MB encoded == 7.5 MB of raw audio
    assert provider.max_audio_bytes == MIMO_MAX_AUDIO_BYTES
    assert provider.max_audio_bytes < MAX_AUDIO_SIZE_BYTES


def test_api_style_values() -> None:
    """The API styles are stable string values."""
    assert ApiStyle.TRANSCRIPTIONS == "transcriptions"
    assert ApiStyle.CHAT_COMPLETIONS == "chat_completions"


def test_custom_transcription_url_ignores_path_rules() -> None:
    """Custom providers always resolve through the URL rules."""
    assert custom_provider("http://host:8000/asr", "m").transcription_url == (
        "http://host:8000/asr"
    )
    assert CONF_API_KEY == "api_key"
