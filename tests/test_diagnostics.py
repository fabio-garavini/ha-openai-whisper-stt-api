"""Tests for diagnostics."""

from __future__ import annotations

from typing import Any

from custom_components.openai_whisper_cloud.const import (
    CONF_BASE_URL,
    CONF_PROVIDER_TYPE,
    DOMAIN,
)
from custom_components.openai_whisper_cloud.diagnostics import (
    async_get_config_entry_diagnostics,
)

from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME

from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_diagnostics_redacts_api_key(hass: Any) -> None:
    """The API key never leaves the instance."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={
            CONF_PROVIDER_TYPE: "custom",
            CONF_NAME: "Local",
            CONF_BASE_URL: "http://127.0.0.1:8000",
            CONF_API_KEY: "super-secret",
        },
        options={CONF_MODEL: "whisper-1", "temperature": 0.0, "prompt": "hello"},
    )
    entry.add_to_hass(hass)
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    data = diagnostics["entry"]["data"]
    assert data[CONF_API_KEY] == "**REDACTED**"
    assert data[CONF_BASE_URL] == "http://127.0.0.1:8000"
    assert data[CONF_NAME] == "Local"
    assert diagnostics["entry"]["options"] == {
        CONF_MODEL: "whisper-1",
        "temperature": 0.0,
        "prompt": "hello",
    }
    # nothing unexpected leaks
    assert "super-secret" not in str(diagnostics)
