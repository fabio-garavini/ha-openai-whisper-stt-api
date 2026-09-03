# Whisper STT Cloud API integration for Home Assistant 🏠🎙️

This HA custom integration lets you use any compatible OpenAI API (OpenAI, GroqCloud, Mistral AI, others coming ...) for computing speech-to-text in cloud, reducing workload on Home Assistant server.

## Sources

- *OpenAI*
- *GroqCloud*
- *Mistral AI*
- *Custom*

## OpenAI

### Requirements 📖

- An OpenAI account 👤  --> You can create one [here](https://platform.openai.com/signup)
- An `API Key` 🔑 --> You can generate one [here](https://platform.openai.com/api-keys)

### Models

- `gpt-4o-mini-transcribe`, `gpt-4o-transcribe` - [Next generation](https://openai.com/index/introducing-our-next-generation-audio-models) OpenAI transcribe models
- `whisper-1` - Despite the name this is the *whisper-large-v2* model

## GroqCloud

### Requirements 📖

- An GroqCloud account 👤  --> You can create one [here](https://console.groq.com/login)
- An `API Key` 🔑 --> You can generate one [here](https://console.groq.com/keys)

### Models

Currently all GroqCloud Whisper models are free up to 28800 audio seconds per day!

- `whisper-large-v3`
- `whisper-large-v3-turbo` - faster version of *whisper-large-v3*

## Mistral AI

### Requirements 📖

- An Mistralai account 👤  --> You can create one [here](https://auth.mistral.ai/ui/registration)
- An `API Key` 🔑 --> You can generate one [here](https://console.mistral.ai/api-keys)

### Models

Currently all Mistral AI models are free up to 1 billion token per month !

- `voxtral-mini`

## Custom

Any OpenAI-compatible transcription API: llama.cpp server, Speaches, LocalAI, faster-whisper-server, whisper.cpp, [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice), and many more.

### Configuration 📖

- `url`: your server URL. Rules are predictable — a URL without a path gets the standard endpoint appended, anything else is used **as-is**:
  - `https://your-host` → uses `https://your-host/v1/audio/transcriptions`
  - `https://your-host/v1` → uses `https://your-host/v1/audio/transcriptions`
  - `https://your-host/any/prefix/v1/audio/transcriptions` → used as-is
  - `http://your-host:8000/asr` → used as-is (whisper-asr-webservice)
- `api_key`: (Optional) leave empty if your server does not require authentication
- `model`: (Required) any model name accepted by your server, e.g. `whisper-1`, `Systran/faster-whisper-large-v3`
- `file_field`: (Optional, advanced) multipart field name used to upload the audio. Most OpenAI-compatible servers use `file`; whisper-asr-webservice needs `audio_file`. If unsure, keep the default `file` — the integration detects the expected field from a `422` response and retries automatically.

### whisper-asr-webservice 🐳

Works out of the box: set the URL to `http://your-host:8000/asr`, leave the API key empty, and pick any model name (e.g. `distil-medium.en`). The plain-text responses of its `/asr` endpoint and its `audio_file` multipart field are handled automatically.

The setup performs a lenient connectivity check: servers that don't implement the `/v1/models` endpoint are still accepted, since only the transcription endpoint matters.

## Upgrading from v1.x ⬆️

Existing v1.x config entries are migrated automatically on restart. Migration detects the old entry format by its shape (not just the version number), so entries affected by the historical v1.3 version-bump bug are migrated correctly too:

- The provider index is converted to a stable provider key
- The selected model index is resolved to the model name
- The provider, model, temperature and prompt settings are preserved

## How to install ⚙️

Before configuring the integration you must first install the `custom_integration`. You can do it through HACS or manually

### HACS ✨

1. **Add** ➕ [this repository](https://my.home-assistant.io/redirect/hacs_repository/?owner=fabio-garavini&repository=ha-openai-whisper-stt-api&category=integration) to your HACS repositories:

    - **Click** on this link ⤵️

      [![Add Repository to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=fabio-garavini&repository=ha-openai-whisper-stt-api&category=integration)

    - Or **copy** this url ⤵️ and paste into your HACS custom repostories

      ```url
      https://github.com/fabio-garavini/ha-openai-whisper-stt-api
      ```

2. **Install** 💻 the `OpenAI Whisper Cloud` integration
3. **Restart** 🔁 Home Assistant

### Manual Install ⌨️

1. **Download** this repository
2. **Copy** everything inside the `custom_components` folder into your Home Assistant's `custom_components` folder.
3. **Restart** Home Assistant

## Configuration 🔧

These are the parameters that you can configure:

- `api_key`: (Required) api key
- `model`: (Required) Check your source API
- `temperature`: (Optional) Sampling temperature between 0 and 1. Default `0`
- `prompt`: (Optional) Can be used to **improve speech recognition** of words or even names. Default `""`
  <br>You have to provide a list of words or names separated by a comma `, `
  <br>Example: `"open, close, Chat GPT-3, DALL·E"`.

Now you can set it up through your Home Assistant Dashboard (YAML configuration not supported).

### Home Assistant Dashboard 💻

- Configure the integration by **clicking here** ⤵️

  [![Add Repository to HACS](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=openai_whisper_cloud)

- Or navigate to your `Devices & services` page and click `+ Add Integration`
