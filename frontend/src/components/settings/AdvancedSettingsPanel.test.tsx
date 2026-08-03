import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AppConfig } from '../../types'
import AdvancedSettingsPanel from './AdvancedSettingsPanel'
import * as client from '../../api/client'

vi.mock('../../api/client', () => ({
  getHealth: vi.fn(),
  resetConfigSection: vi.fn(),
  saveConfig: vi.fn(),
}))

const config = {
  language: 'en',
  log_level: 'INFO',
  stt: {
    sherpa_model_id: 'auto',
    sherpa_onnx_provider: 'cpu',
    process_interval_s: 2,
    max_phrase_s: 10,
    blocked_phrases: ['Дима Торжок'],
    sherpa_blank_penalty: null,
  },
  translation: {
    model_path: 'models/e4b.gguf',
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
    dfn2: { enabled: true, stt_atten_lim_db: 20 },
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

describe('AdvancedSettingsPanel', () => {
  beforeEach(() => {
    vi.mocked(client.saveConfig).mockImplementation(async value => value)
    vi.mocked(client.resetConfigSection).mockResolvedValue(config)
    vi.mocked(client.getHealth).mockResolvedValue({
      ok: false,
      checks: [
        { name: 'python', ok: true, detail: '3.12' },
        { name: 'translation_model', ok: false, detail: 'missing' },
      ],
    })
  })

  it('keeps Health visible while the editor is hidden and reports progress', async () => {
    render(
      <AdvancedSettingsPanel
        config={config}
        showEditor={false}
        onConfigChange={vi.fn()}
      />,
    )

    expect(screen.queryByText('Advanced config')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Run' }))
    expect(screen.getByRole('button', { name: 'Running...' })).toBeDisabled()
    expect(await screen.findByText('Some checks need attention')).toBeInTheDocument()
    expect(screen.getByText('translation_model')).toBeInTheDocument()
  })

  it('edits and saves MTP settings without changing the config structure', async () => {
    const onConfigChange = vi.fn()
    render(
      <AdvancedSettingsPanel
        config={config}
        showEditor
        onConfigChange={onConfigChange}
      />,
    )

    fireEvent.click(screen.getByRole('tab', { name: 'Translation' }))
    const mtpN = await screen.findByLabelText('mtp_n')
    fireEvent.change(mtpN, { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Translation' }))

    await waitFor(() => expect(client.saveConfig).toHaveBeenCalledTimes(1))
    const saved = vi.mocked(client.saveConfig).mock.calls[0][0]
    expect(saved.translation.mtp_n).toBe(4)
    expect(saved.translation.mtp_enabled).toBe(true)
    expect(Object.keys(saved.translation)).toEqual(Object.keys(config.translation))
    expect(onConfigChange).toHaveBeenCalledWith(saved)
  })

  it('parses a nullable advanced numeric value without turning null into text', async () => {
    render(
      <AdvancedSettingsPanel
        config={config}
        showEditor
        onConfigChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('tab', { name: 'STT' }))
    const penalty = await screen.findByLabelText('sherpa_blank_penalty')
    fireEvent.change(penalty, { target: { value: '0.3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save STT' }))

    await waitFor(() => expect(client.saveConfig).toHaveBeenCalledTimes(1))
    expect(
      vi.mocked(client.saveConfig).mock.calls[0][0].stt.sherpa_blank_penalty,
    ).toBe(0.3)
  })

  it('requires confirmation before resetting a section', async () => {
    render(
      <AdvancedSettingsPanel
        config={config}
        showEditor
        onConfigChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Reset Audio' }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(client.resetConfigSection).not.toHaveBeenCalled()
  })
})
