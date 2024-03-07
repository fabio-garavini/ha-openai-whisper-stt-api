# OpenAI Whisper STT Cloud API integration for Home Assistant

This is a custom integration for using OpenAI cloud speech-to-text API in the Assist pipeline. Reducing the workload on the Home Assistant server.

## Requirements:
- An OpenAI account and `API Key`

## How to install
### HACS
Add [this](https://github.com/fabio-garavini/ha-openai-whisper-stt-api) repository to your HACS repositories and then search and install the `OpenAI Whisper Cloud` integration
```
https://github.com/fabio-garavini/ha-openai-whisper-stt-api
```

### Manual
Download this repository and copy everything inside the `custom_components` folder inside your Home Assistant's `custom_components` folder.

## Configuration

- `api_key`: (Required) OpenAI api key
- `model`: (Required) At the moment the only model available is `whisper-1`
- `temperature`: (Optional) Sampling temperature between 0 and 1. Default `0`
- `prompt`: (Optional) Used to guide the model's style. Default ` `

### HA Interface
Add the integration from the Devices & services page

### Manual
Add in your `configuration.yaml` this:
```
stt:
  - platform: openai_whisper_cloud
    api_key: <your-api-key>
    model: "whisper-1"
    temperature: 0
    prompt: ""
```