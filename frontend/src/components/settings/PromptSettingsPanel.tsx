import { useEffect, useState } from 'react'
import type { Capabilities, PromptPreset, PromptState } from '../../types'
import { ConfirmDialog } from './Dialog'
import SelectField from './SelectField'
import { promptPresetLabel } from './promptLabels'
import { LANGUAGE_OPTIONS, languageDisplayName, languageOption } from './languages'

type Props = {
  prompts: PromptState | null
  onChange: (next: PromptState) => void
  onSave: (next?: PromptState) => void | Promise<void>
  onReset: (id?: string) => void | Promise<void>
  onDelete: (id: string) => void | Promise<void>
  capabilities: Capabilities | null
}

export default function PromptSettingsPanel({ prompts, onChange, onSave, onReset, onDelete, capabilities }: Props) {
  const [isEditing, setIsEditing] = useState(false)
  const [draftBeforeEdit, setDraftBeforeEdit] = useState<PromptState | null>(null)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [resetDialogOpen, setResetDialogOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [validationError, setValidationError] = useState('')
  if (!prompts) return <div className="panel subpanel">Loading prompts...</div>

  const activePreset = prompts.presets[prompts.active]
  const canResetActive = Boolean(prompts.factory_ids?.includes(prompts.active))
  const promptOptions = Object.entries(prompts.presets).map(([id, preset]) => ({
    label: promptPresetLabel(preset),
    value: id,
  }))
  const capabilityLanguages = capabilities?.languages ?? LANGUAGE_OPTIONS
  const sourceOptions = capabilityLanguages.filter(option => capabilities?.source_languages.includes(option.value) ?? true)
  const targetOptions = capabilityLanguages.filter(option => capabilities?.target_languages.includes(option.value) ?? true)
  const activeSource = capabilityLanguageCode(activePreset?.source_language ?? 'en', capabilityLanguages)
  const activeTarget = capabilityLanguageCode(activePreset?.target_language ?? 'ru', capabilityLanguages)
  const activeModelWarning = compatibilityWarning(activeSource, activeTarget, capabilities)

  const updateActivePreset = (patch: Partial<PromptPreset>) => {
    if (!isEditing || !activePreset) return
    if (validationError) setValidationError('')
    onChange({
      ...prompts,
      presets: {
        ...prompts.presets,
        [prompts.active]: {
          ...activePreset,
          ...patch,
        },
      },
    })
  }

  const beginEdit = () => {
    setValidationError('')
    setDraftBeforeEdit(prompts)
    setIsEditing(true)
  }

  const cancelEdit = () => {
    if (draftBeforeEdit) onChange(draftBeforeEdit)
    setValidationError('')
    setDraftBeforeEdit(null)
    setIsEditing(false)
    console.debug('[FIX:prompt-validation] cleared validation state on edit cancel')
  }

  const createPrompt = async (sourceLanguage: string, targetLanguage: string, customName: string) => {
    const sourceName = languageDisplayName(sourceLanguage)
    const targetName = languageDisplayName(targetLanguage)
    const name = customName.trim() || `${sourceName} to ${targetName}`
    if (hasPromptName(prompts, name)) {
      setValidationError(`A preset named “${name}” already exists.`)
      return
    }
    const id = uniquePromptId(name, Object.keys(prompts.presets))
    const next = {
      ...prompts,
      active: id,
      presets: {
        ...prompts.presets,
        [id]: {
          name,
          source_language: sourceLanguage,
          target_language: targetLanguage,
          template: buildPromptTemplate(sourceName, targetName),
        },
      },
    }
    try {
      await onSave(next)
      setCreateDialogOpen(false)
      setValidationError('')
      setDraftBeforeEdit(null)
      setIsEditing(false)
    } catch (error) {
      setValidationError(promptSaveError(error))
      console.debug('[prompts.create] failed', error)
    }
  }

  const savePrompt = async () => {
    const name = activePreset?.name.trim() ?? ''
    if (!name || hasPromptName(prompts, name, prompts.active)) {
      setValidationError(!name ? 'Preset name is required.' : `A preset named “${name}” already exists.`)
      return
    }
    try {
      await onSave()
      setValidationError('')
      setDraftBeforeEdit(null)
      setIsEditing(false)
    } catch (error) {
      setValidationError(promptSaveError(error))
      console.debug('[prompts.save] failed', error)
    }
  }

  const deleteActivePrompt = async () => {
    try {
      await onDelete(prompts.active)
      setValidationError('')
      setDeleteDialogOpen(false)
      setDraftBeforeEdit(null)
      setIsEditing(false)
    } catch (error) {
      setDeleteDialogOpen(false)
      setValidationError('Could not delete this preset. Reload the backend and try again.')
      console.debug('[prompts.delete] failed', error)
    }
  }

  const resetPrompt = async () => {
    await onReset(prompts.active)
    setResetDialogOpen(false)
    setDraftBeforeEdit(null)
    setIsEditing(false)
  }

  return (
    <div className="panel subpanel prompt-panel">
      <div className="section-head compact">
        <div>
          <h2>Prompts</h2>
          <p>Prompt text is read-only until you explicitly edit it.</p>
        </div>
        <div className="inline-actions">
          {!isEditing && !canResetActive && <button className="btn small danger" onClick={() => setDeleteDialogOpen(true)}>Delete</button>}
          {!isEditing && <button className="btn small" onClick={() => { setValidationError(''); setCreateDialogOpen(true) }}>Create</button>}
          {!isEditing && <button className="btn small" onClick={beginEdit}>Edit current</button>}
          {isEditing && canResetActive && <button className="btn small danger" onClick={() => setResetDialogOpen(true)}>Reset</button>}
          {isEditing && <button className="btn small" onClick={cancelEdit}>Cancel</button>}
          {isEditing && <button className="btn small primary" onClick={savePrompt}>Save</button>}
        </div>
      </div>
      {validationError && !createDialogOpen && <div className="prompt-validation-error">{validationError}</div>}
      {isEditing ? (
        <label className="field">
          <span>Preset</span>
          <input
            value={activePreset?.name ?? ''}
            onChange={event => updateActivePreset({ name: event.target.value })}
          />
        </label>
      ) : (
        <SelectField
          label="Preset"
          value={prompts.active}
          options={promptOptions}
          searchable
          searchPlaceholder="Search preset..."
          onChange={value => {
            setIsEditing(false)
            setValidationError('')
            onChange({ ...prompts, active: value })
          }}
        />
      )}
      <div className="prompt-meta-grid">
        {isEditing ? (
          <SelectField
            label="Prompt source"
            value={activeSource}
            options={sourceOptions}
            searchable
            searchPlaceholder="Search STT language..."
            onChange={value => updateActivePreset({ source_language: value })}
          />
        ) : (
          <div className="field readonly-field">
            <span>Prompt source</span>
            <strong>{languageOption(activePreset?.source_language ?? '').label}</strong>
          </div>
        )}

        {isEditing ? (
          <SelectField
            label="Prompt target"
            value={activeTarget}
            options={targetOptions}
            searchable
            searchPlaceholder="Search translation language..."
            onChange={value => updateActivePreset({ target_language: value })}
          />
        ) : (
          <div className="field readonly-field">
            <span>Prompt target</span>
            <strong>{languageOption(activeTarget).label}</strong>
          </div>
        )}
      </div>
      {activeModelWarning && <div className="model-compatibility-warning">{activeModelWarning}</div>}
      <label className="field prompt-editor-field">
        <span>System prompt</span>
        <textarea
          className={`prompt-editor ${isEditing ? 'editing' : 'readonly'}`}
          rows={9}
          value={activePreset?.template ?? ''}
          readOnly={!isEditing}
          aria-disabled={!isEditing}
          onChange={event => updateActivePreset({ template: event.target.value })}
        />
      </label>
      <CreatePromptDialog
        open={createDialogOpen}
        onConfirm={createPrompt}
        onCancel={() => { setCreateDialogOpen(false); setValidationError('') }}
        onClearError={() => setValidationError('')}
        error={validationError}
        capabilities={capabilities}
      />
      <ConfirmDialog
        open={resetDialogOpen}
        title="Reset prompt preset?"
        message="This will restore the factory prompt text and metadata for this preset. Unsaved edits will be lost."
        confirmLabel="Reset"
        danger
        onConfirm={resetPrompt}
        onCancel={() => setResetDialogOpen(false)}
      />
      <ConfirmDialog
        open={deleteDialogOpen}
        title="Delete prompt preset?"
        message="This custom preset will be permanently removed. Factory presets cannot be deleted."
        confirmLabel="Delete"
        danger
        onConfirm={deleteActivePrompt}
        onCancel={() => setDeleteDialogOpen(false)}
      />
    </div>
  )
}

function CreatePromptDialog({
  open,
  error,
  onConfirm,
  onCancel,
  onClearError,
  capabilities,
}: {
  open: boolean
  error: string
  onConfirm: (source: string, target: string, customName: string) => void
  onCancel: () => void
  onClearError: () => void
  capabilities: Capabilities | null
}) {
  const [source, setSource] = useState('en')
  const [target, setTarget] = useState('ru')
  const [useCustomName, setUseCustomName] = useState(false)
  const [customName, setCustomName] = useState('')

  useEffect(() => {
    if (!open) return
    setSource('en')
    setTarget('ru')
    setUseCustomName(false)
    setCustomName('')
  }, [open])

  if (!open) return null
  const defaultName = `${languageDisplayName(source)} to ${languageDisplayName(target)}`
  const canCreate = !useCustomName || Boolean(customName.trim())
  const languageOptions = capabilities?.languages ?? LANGUAGE_OPTIONS
  const sourceOptions = languageOptions.filter(option => capabilities?.source_languages.includes(option.value) ?? true)
  const targetOptions = languageOptions.filter(option => capabilities?.target_languages.includes(option.value) ?? true)
  const modelWarning = compatibilityWarning(source, target, capabilities)
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div className="dialog-card prompt-create-dialog" role="dialog" aria-modal="true" onMouseDown={event => event.stopPropagation()}>
        <h3>Create prompt preset</h3>
        <div className="prompt-meta-grid">
          <SelectField label="From" value={source} options={sourceOptions} searchable onChange={value => { onClearError(); setSource(value) }} />
          <SelectField label="To" value={target} options={targetOptions} searchable onChange={value => { onClearError(); setTarget(value) }} />
        </div>
        {modelWarning && <div className="model-compatibility-warning">{modelWarning}</div>}
        <div className="default-preset-name">Default name: <strong>{defaultName}</strong></div>
        <label className="check-field">
          <input type="checkbox" checked={useCustomName} onChange={event => { onClearError(); setUseCustomName(event.target.checked) }} />
          <span>Use custom name</span>
        </label>
        {useCustomName && (
          <label className="field">
            <span>Custom name</span>
            <input autoFocus value={customName} onChange={event => { onClearError(); setCustomName(event.target.value) }} />
          </label>
        )}
        {error && <div className="prompt-validation-error">{error}</div>}
        <div className="dialog-actions">
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn primary" disabled={!canCreate} onClick={() => onConfirm(source, target, useCustomName ? customName : '')}>Create</button>
        </div>
      </div>
    </div>
  )
}

function hasPromptName(prompts: PromptState, name: string, exceptId = ''): boolean {
  const normalized = name.trim().toLocaleLowerCase()
  return Object.entries(prompts.presets).some(([id, preset]) => (
    id !== exceptId && promptPresetLabel(preset).trim().toLocaleLowerCase() === normalized
  ))
}

function buildPromptTemplate(source: string, target: string): string {
  return `TASK: Translate ${source} subtitles into ${target}.\nSTYLE: friendly.\nTranslate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. Return only the final translation without labels or commentary.`
}


function capabilityLanguageCode(value: string, options: { label: string; value: string }[]): string {
  const normalized = value.trim().toLocaleLowerCase()
  return options.find(option => (
    option.value.toLocaleLowerCase() === normalized
    || option.label.replace(/\s*\([^)]*\)\s*$/, '').toLocaleLowerCase() === normalized
  ))?.value ?? normalized
}

function compatibilityWarning(source: string, target: string, capabilities: Capabilities | null): string {
  if (!capabilities) return ''
  const compatible = capabilities.translation_models.filter(model => (
    model.languages.includes(source) && model.languages.includes(target)
  ))
  if (!compatible.length) return 'No installed translation model supports this language pair.'
  return ''
}

function uniquePromptId(name: string, used: string[]): string {
  const base = `custom_${name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'prompt'}`
  const usedSet = new Set(used)
  if (!usedSet.has(base)) return base
  let index = 2
  while (usedSet.has(`${base}_${index}`)) {
    index += 1
  }
  return `${base}_${index}`
}

function promptSaveError(error: unknown): string {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response
    if (typeof response?.data?.detail === 'string' && response.data.detail.trim()) {
      return response.data.detail
    }
  }
  return 'Could not save this preset. Please try again.'
}
