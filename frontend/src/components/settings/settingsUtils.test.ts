import { describe, expect, it } from 'vitest'
import type { AppConfig, Capabilities } from '../../types'
import {
  capabilityLanguageCode,
  configsEqual,
  contextWindowForSubtitles,
  sameModelFile,
  withCustomOption,
} from './settingsUtils'

const config = {
  language: 'en',
  log_level: 'INFO',
  stt: {
    sherpa_model_id: 'auto',
    sherpa_onnx_provider: 'cpu',
    process_interval_s: 2,
    max_phrase_s: 10,
  },
  translation: {
    model_path: 'models/model.gguf',
    device: 'gpu',
    n_gpu_layers: -1,
    mtp_enabled: true,
    mtp_model_path: 'auto',
    mtp_n: 1,
    llama_server_path: 'llama-server',
    n_ctx: 1024,
    n_batch: 256,
    context_subtitles: 3,
    target_language: 'Russian',
    prompt_preset: 'en_to_ru',
  },
  audio: {
    capture_mode: 'browser_tab',
    device_name: '',
    input_gain: 1,
    dfn2: {},
  },
  overlay: {
    enabled: true,
    mode: 'browser',
    opacity: 0.82,
    font_size: 23,
    source_font_size: 15,
    show_source: true,
  },
} satisfies AppConfig

const capabilities = {
  languages: [
    { value: 'en', label: 'English (en)' },
    { value: 'ru', label: 'Russian (ru)' },
  ],
  source_languages: ['en'],
  target_languages: ['ru'],
  stt_models: [],
  stt_defaults: {},
  stt_providers: [],
  translation_models: [],
} satisfies Capabilities

describe('settings helpers', () => {
  it('ignores server metadata when comparing autosave snapshots', () => {
    expect(configsEqual(
      { ...config, _hash: 'old' },
      { ...config, _hash: 'new' },
    )).toBe(true)
    expect(configsEqual(
      config,
      { ...config, overlay: { ...config.overlay, opacity: 0.5 } },
    )).toBe(false)
  })

  it('normalizes full language labels and codes', () => {
    expect(capabilityLanguageCode('English', capabilities)).toBe('en')
    expect(capabilityLanguageCode('RU', capabilities)).toBe('ru')
    expect(capabilityLanguageCode('Unknown', capabilities)).toBe('unknown')
  })

  it('matches built-in models by filename across absolute paths', () => {
    expect(sameModelFile(
      'models/Translate.GGUF',
      'C:\\models\\translate.gguf',
    )).toBe(true)
    expect(sameModelFile('', '')).toBe(false)
  })

  it.each([
    [-1, 544],
    [0, 544],
    [3, 1024],
    [5, 1344],
    [99, 1344],
    [2.9, 864],
  ])('maps context count %s to safe n_ctx %s', (count, expected) => {
    expect(contextWindowForSubtitles(count)).toBe(expected)
  })

  it('adds exactly one Custom option only for unknown values', () => {
    const options = [{ label: 'CPU', value: 'cpu' }]
    expect(withCustomOption(options, 'cpu')).toBe(options)
    expect(withCustomOption(options, 'directml')).toEqual([
      { label: 'Custom', value: 'directml', badge: 'directml' },
      ...options,
    ])
  })
})
