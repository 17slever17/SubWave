import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { PromptState } from '../../types'
import PromptSettingsPanel from './PromptSettingsPanel'

const prompts: PromptState = {
  active: 'custom_spanish',
  factory_ids: ['ja_to_en'],
  presets: {
    ja_to_en: {
      name: 'Japanese to English',
      source_language: 'ja',
      target_language: 'English',
      template: 'Translate Japanese subtitles into English.',
    },
    custom_spanish: {
      name: 'Shiro1213312',
      source_language: 'es',
      target_language: 'Russian',
      template: 'Translate Spanish subtitles into Russian.',
    },
  },
}

describe('PromptSettingsPanel', () => {
  it('shows the actual backend error when saving fails', async () => {
    const onSave = vi.fn().mockRejectedValue({
      response: { data: { detail: "No STT model supports 'xx'" } },
    })
    render(
      <PromptSettingsPanel
        prompts={structuredClone(prompts)}
        onChange={vi.fn()}
        onSave={onSave}
        onReset={vi.fn()}
        onDelete={vi.fn()}
        capabilities={null}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit current' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(screen.getByText("No STT model supports 'xx'")).toBeInTheDocument())
  })
})
