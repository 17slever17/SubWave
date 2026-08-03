import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AppConfig, Capabilities, PromptState } from '../../types'
import SettingsPage from './SettingsPage'
import * as client from '../../api/client'

vi.mock('../../api/client', () => ({
  applyPreset: vi.fn(),
  deletePrompt: vi.fn(),
  getCapabilities: vi.fn(),
  getAudioDevices: vi.fn(),
  getConfig: vi.fn(),
  getModelStatus: vi.fn(),
  getPresets: vi.fn(),
  getPrompts: vi.fn(),
  previewOverlayConfig: vi.fn(),
  saveConfig: vi.fn(),
  savePrompts: vi.fn(),
  resetPrompt: vi.fn(),
  getHealth: vi.fn(),
  resetConfigSection: vi.fn(),
}))

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
    model_path: 'models/translate_gemma4_sub-E4B-Q4_K_XL.gguf',
    device: 'gpu',
    n_gpu_layers: -1,
    mtp_enabled: true,
    mtp_model_path: 'auto',
    mtp_n: 1,
    llama_server_path: 'bin/llama-server.exe',
    n_ctx: 1024,
    n_batch: 256,
    context_subtitles: 3,
    target_language: 'Russian',
    prompt_preset: 'en_to_ru',
  },
  audio: {
    capture_mode: 'browser_tab',
    device_name: 'CABLE Output',
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

const prompts = {
  active: 'en_to_ru',
  factory_ids: ['en_to_ru'],
  presets: {
    en_to_ru: {
      name: 'English to Russian',
      source_language: 'en',
      target_language: 'Russian',
      template: 'Translate.',
    },
  },
} satisfies PromptState

const capabilities = {
  languages: [
    { value: 'en', label: 'English (en)' },
    { value: 'ru', label: 'Russian (ru)' },
  ],
  source_languages: ['en'],
  target_languages: ['ru'],
  stt_models: [
    { value: 'parakeet-en', label: 'Parakeet English', languages: ['en'], installed: false },
  ],
  stt_defaults: { en: 'parakeet-en' },
  stt_providers: [{ value: 'cpu', label: 'CPU (Recommended)' }],
  translation_models: [
    {
      value: 'models/translate_gemma4_sub-E4B-Q4_K_XL.gguf',
      label: 'Translate Gemma Sub E4B Q4 (better)',
      languages: ['en', 'ru'],
    },
    {
      value: 'models/translate_gemma4_sub-E2B-Q4_K_XL.gguf',
      label: 'Translate Gemma Sub E2B Q4 (faster)',
      languages: ['en', 'ru'],
    },
  ],
} satisfies Capabilities

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(client.getConfig).mockResolvedValue(structuredClone(config))
    vi.mocked(client.getPresets).mockResolvedValue([])
    vi.mocked(client.getPrompts).mockResolvedValue(structuredClone(prompts))
    vi.mocked(client.getAudioDevices).mockResolvedValue([])
    vi.mocked(client.getModelStatus).mockResolvedValue({
      stt: { path: 'stt', exists: true },
      translation: { path: 'llm', exists: true },
    })
    vi.mocked(client.getCapabilities).mockResolvedValue(structuredClone(capabilities))
    vi.mocked(client.previewOverlayConfig).mockResolvedValue({ ok: true })
    vi.mocked(client.savePrompts).mockImplementation(async value => structuredClone(value))
    vi.mocked(client.saveConfig).mockImplementation(async value => ({
      ...structuredClone(value),
      _hash: 'saved',
    }))
    vi.mocked(client.getHealth).mockResolvedValue({ ok: true, checks: [] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('loads settings and autosaves a font change with matching source size', async () => {
    render(<SettingsPage />)
    expect(screen.queryByLabelText('Audio source')).not.toBeInTheDocument()
    const input = await screen.findByLabelText('Subtitle font size')

    fireEvent.change(input, { target: { value: '30' } })

    await waitFor(() => expect(client.saveConfig).toHaveBeenCalledTimes(1), {
      timeout: 1500,
    })
    const saved = vi.mocked(client.saveConfig).mock.calls[0][0]
    expect(saved.overlay.font_size).toBe(30)
    expect(saved.overlay.source_font_size).toBe(20)
  })

  it('offers a compatible ASR model before it has been downloaded', async () => {
    render(<SettingsPage />)

    expect(await screen.findByRole('button', { name: 'ASR model: Parakeet English' })).toBeInTheDocument()
  })

  it('shows E4B as the default translation model with inline quality labels', async () => {
    render(<SettingsPage />)

    const selector = await screen.findByRole('button', {
      name: 'Translation model: Translate Gemma Sub E4B Q4 (better)',
    })
    fireEvent.click(selector)

    expect(screen.getByRole('option', {
      name: 'Translate Gemma Sub E2B Q4 (faster)',
    })).toBeInTheDocument()
    expect(within(selector.parentElement!).queryByText('Recommended')).not.toBeInTheDocument()
  })

  it('previews overlay changes before the autosave delay', async () => {
    render(<SettingsPage />)
    const checkbox = await screen.findByRole('checkbox', { name: 'Show previous line' })

    fireEvent.click(checkbox)

    await waitFor(() => expect(client.previewOverlayConfig).toHaveBeenCalledWith(
      expect.objectContaining({ show_source: false }),
    ))
  })

  it('shows custom model and device values instead of silently replacing them', async () => {
    vi.mocked(client.getConfig).mockResolvedValue({
      ...structuredClone(config),
      stt: { ...config.stt, sherpa_onnx_provider: 'directml' },
      translation: {
        ...config.translation,
        model_path: 'D:\\models\\private.gguf',
      },
    })

    render(<SettingsPage />)

    const model = await screen.findByRole('button', { name: 'Translation model: Custom' })
    const provider = screen.getByRole('button', { name: 'ASR device: Custom' })
    expect(model.parentElement).toHaveTextContent('private.gguf')
    expect(provider.parentElement).toHaveTextContent('directml')
  })

  it('persists a newly created prompt immediately without entering edit mode', async () => {
    render(<SettingsPage />)
    await screen.findByText('Prompts')
    vi.mocked(client.saveConfig).mockClear()

    fireEvent.click(screen.getByRole('button', { name: 'Create' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Use custom name' }))
    fireEvent.change(screen.getByLabelText('Custom name'), {
      target: { value: 'Random' },
    })
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Create' }))

    await waitFor(() => expect(client.savePrompts).toHaveBeenCalledTimes(1))
    const saved = vi.mocked(client.savePrompts).mock.calls[0][0]
    expect(saved.active).toBe('custom_random')
    expect(saved.presets.custom_random.name).toBe('Random')
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument()
    await waitFor(() => expect(client.saveConfig).toHaveBeenCalledTimes(1), {
      timeout: 1500,
    })
  })

  it('repairs a runtime config that references a missing prompt preset', async () => {
    vi.mocked(client.getConfig).mockResolvedValue({
      ...structuredClone(config),
      language: 'ja',
      translation: {
        ...config.translation,
        prompt_preset: 'custom_missing',
        target_language: 'Russian',
      },
    })

    render(<SettingsPage />)

    await waitFor(() => expect(client.saveConfig).toHaveBeenCalledTimes(1), {
      timeout: 1500,
    })
    const repaired = vi.mocked(client.saveConfig).mock.calls[0][0]
    expect(repaired.translation.prompt_preset).toBe('en_to_ru')
    expect(repaired.language).toBe('en')
    expect(repaired.translation.target_language).toBe('Russian')
  })
})
