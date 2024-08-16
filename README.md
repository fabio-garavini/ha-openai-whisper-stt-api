# OpenAI Whisper STT Cloud API integration for Home Assistant 🏠🎙️

This is a custom integration for using OpenAI cloud speech-to-text API in the Assist pipeline, reducing the workload on the Home Assistant server.

## Requirements 📖

- An OpenAI account 👤  --> You can create one [here](https://platform.openai.com/signup)
- An `API Key` 🔑 --> You can generate one [here](https://platform.openai.com/api-keys)

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

- `api_key`: (Required) OpenAI api key
- `model`: (Required) At the moment the only model available is `whisper-1` which actually is the *whisper-large-v2* model
- `temperature`: (Optional) Sampling temperature between 0 and 1. Default `0`
- `prompt`: (Optional) Can be used to **improve speech recognition** of words or even names. Default `""`
  <br>You have to provide a list of words or names separated by a comma `, `
  <br>Example: `"open, close, Chat GPT-3, DALL·E"`.

Now you can set it up through your Home Assistant Dashboard (YAML configuration not supported).

### Home Assistant Dashboard 💻

- Configure the integration by **clicking here** ⤵️

  [![Add Repository to HACS](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=openai_whisper_cloud)

- Or navigate to your `Devices & services` page and click `+ Add Integration`
