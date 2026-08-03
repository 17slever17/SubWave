import type { PromptPreset } from '../../types'
import { languageDisplayName } from './languages'

export function promptDirectionLabel(preset: PromptPreset): string {
  const source = languageDisplayName(preset.source_language)
  const target = languageDisplayName(preset.target_language) || 'Unknown'
  return `${source} to ${target}`
}

export function promptPresetLabel(preset: PromptPreset): string {
  const direction = promptDirectionLabel(preset)
  const name = preset.name.trim()
  if (!name || normalizeLabel(name) === normalizeLabel(direction)) {
    return direction
  }
  return `${name} (${direction})`
}

function normalizeLabel(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}
