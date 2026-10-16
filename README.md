# Whisper STT Cloud API integration for Home Assistant 🏠🎙️

This HA custom integration lets you use any compatible OpenAI API (OpenAI, GroqCloud, Mistral AI, others coming ...) for computing speech-to-text in cloud, reducing workload on Home Assistant server.

## Sources

- *OpenAI*
- *GroqCloud*
- *Mistral AI*
- *Cohere*
- *Xiaomi MiMo*
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

## Cohere

### Requirements 📖

- A Cohere account and an API key from the [Cohere dashboard](https://dashboard.cohere.com/api-keys)

### Models

- `cohere-transcribe-03-2026` — served through Cohere's v2 audio API. The API does not accept a prompt, temperature, or streamed uploads, so those options are not shown for this provider.

## Xiaomi MiMo

### Requirements 📖

- A Xiaomi MiMo account 👤 and an `API Key` 🔑 from the [MiMo open platform](https://platform.xiaomimimo.com/)

### Models

- `mimo-v2.5-asr` — bilingual Chinese/English recognition with automatic language detection. It natively recognizes Cantonese, Wu, Minnan, Sichuan and other Chinese dialects, stays stable in noisy environments, far-field pickup and overlapping speakers, and handles specialized content like technical terminology and proper nouns.

**How it works**: unlike the other providers, MiMo serves speech recognition through the OpenAI compatible chat completions API — the audio is sent as a base64 encoded WAV in an `input_audio` content part, and the transcription comes back in `choices[0].message.content`. Because of this:

- the multipart-specific options (prompt, temperature, streamed upload) do not apply and are not shown
- the encoded audio must stay under 10 MB, so recordings are capped at 7.5 MB (about 4 minutes of 16 kHz mono audio)
- the language option only accepts `zh` and `en`; everything else is left to MiMo's automatic detection

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
- `temperature`: (Optional) Sampling temperature between 0 and 1. Default `0`. Not offered for providers that do not support it (e.g. Cohere).
- `prompt`: (Optional) Can be used to **improve speech recognition** of words or even names. Default `""`
  <br>You have to provide a list of words or names separated by a comma `, `
  <br>Example: `"open, close, Chat GPT-3, DALL·E"`.

  The prompt can also be a Home Assistant **template**, which is rendered fresh before *every* transcription. This lets you bias the model with the words that actually exist in your home (entity, area or script names) without maintaining the list by hand, e.g.:

  ```yaml
  {{ states('sensor.room_names') }},{{ states('sensor.entity_names') }},{{ states('sensor.script_names') }}
  ```

  Splitting the names across multiple template sensors is a good way to get around the 255 character limit of a single sensor state. Templates can also build the list directly from your entities, e.g. all light and script names:

  ```yaml
  {{ states.light | map(attribute='name') | join(', ') }},{{ states.script | map(attribute='name') | join(', ') }}
  ```

  Plain text, templates, or a mix of both all work. Because templates are evaluated at transcription time (not at startup), newly added or renamed entities are picked up automatically, and entities that are not yet set up when the integration loads can never break the prompt. The template syntax is validated when you save the settings, and a template that fails to render is logged and skipped instead of failing the transcription.

  **Note**: transcription APIs limit the prompt size (`whisper-1` accepts about 224 tokens, the `gpt-4o-*` models 1024), so keep the generated list focused on the words you actually speak.

- `streaming_upload`: (Optional) **Stream the audio to the provider while it is still being recorded** instead of uploading it after the recording completes. Default: on. See [Streaming uploads](#streaming-uploads-).

## Streaming uploads ⚡

By default the integration does not wait for the end of your voice command to start talking to the provider: audio is uploaded **while it is captured**, so by the time you stop speaking the upload is (almost) already done. This cuts the whole upload time out of the end-to-end latency, which is most noticeable on slow uplinks and with longer commands.

How it stays reliable:

- The live audio is wrapped in a WAV container with unknown sizes (`0xFFFFFFFF`) — the same thing `ffmpeg` writes when streaming WAV to a pipe, so every ffmpeg-based server (OpenAI, GroqCloud, and virtually all self-hosted ones) understands it.
- The audio is simultaneously kept in memory, so if the provider rejects a streamed request the integration transparently retries with a classic buffered upload and stops trying to stream for that session.
- If you prefer the classic behavior, or your server is known not to accept streamed uploads, turn the option off in the integration settings.

Two more runtime behaviors make transcription robust for every provider:

- **Multipart field learning**: if a server answers `422` naming the field it expects the audio in (like whisper-asr-webservice does), the field name is learned and reused, including for streamed requests.
- **Transient error retry**: rate limits (`429`, honoring `Retry-After`) and transient server errors (`500`, `502`, `503`, `504`) are retried once before giving up.

## Diagnostics 🩺

The integration supports Home Assistant's *Download diagnostics* (menu on the integration entry): it reports the entry configuration with the API key redacted, which is safe to attach to a bug report.

## Entity state 📊

Home Assistant defines the state of an STT entity as the **UTC timestamp of the last transcription** (it is `unknown` until the entity is first used, and it is restored across restarts). Home Assistant's assist pipeline skips the framework's own bookkeeping, so this integration maintains it itself: the state updates the moment a transcription starts.

Because that state format is fixed by Home Assistant (`@final` in the base class), live activity is exposed through the entity attributes instead. In *Developer tools → States* you will find:

| Attribute | Meaning |
|---|---|
| `processing` | `true` while a transcription is running |
| `provider` | Provider name (e.g. `OpenAI`, `Custom`) |
| `model` | Model name (e.g. `gpt-4o-mini-transcribe`) |
| `streaming_upload` | Whether uploads are streamed while recording |

This makes it easy to build automations, e.g. trigger on `state` changes (each transcription updates it) or on `processing` turning `true`/`false`.

Now you can set it up through your Home Assistant Dashboard (YAML configuration not supported).

### Home Assistant Dashboard 💻

- Configure the integration by **clicking here** ⤵️

  [![Add Repository to HACS](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=openai_whisper_cloud)

- Or navigate to your `Devices & services` page and click `+ Add Integration`
