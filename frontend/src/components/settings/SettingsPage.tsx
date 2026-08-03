import { memo, useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { Gauge, SlidersHorizontal } from 'lucide-react'
import {
  applyPreset,
  deletePrompt,
  getCapabilities,
  getAudioDevices,
  getConfig,
  getModelStatus,
  getPresets,
  getPrompts,
  previewOverlayConfig,
  saveConfig,
  savePrompts,
  resetPrompt,
} from '../../api/client'
import type { AppConfig, AudioDevice, Capabilities, Preset, PromptState } from '../../types'
import AdvancedSettingsPanel from './AdvancedSettingsPanel'
import { PresetComparisonDialog } from './Dialog'
import PromptSettingsPanel from './PromptSettingsPanel'
import SelectField from './SelectField'
import { promptPresetLabel } from './promptLabels'
import {
  capabilityLanguageCode,
  configsEqual,
  contextWindowForSubtitles,
  modelFileName,
  sameModelFile,
  withCustomOption,
} from './settingsUtils'

const TRANSLATION_DEVICE_OPTIONS = [
  { label: 'GPU', value: 'gpu' },
  { label: 'CPU', value: 'cpu' },
]

const CONTEXT_SUBTITLE_OPTIONS = Array.from({ length: 6 }, (_, count) => ({
  label: String(count),
  value: String(count),
  badge: count === 3 ? 'Recommended' : undefined,
  recommended: count === 3,
}))

export default function SettingsPage() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [draft, setDraft] = useState<AppConfig | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [prompts, setPrompts] = useState<PromptState | null>(null)
  const [devices, setDevices] = useState<AudioDevice[]>([])
  const [modelStatus, setModelStatus] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const [pendingPreset, setPendingPreset] = useState<Preset | null>(null)
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const overlayPreviewTimer = useRef<number | null>(null)

  const pushOverlayPreview = useCallback((overlay: AppConfig['overlay']) => {
    if (overlayPreviewTimer.current !== null) {
      window.clearTimeout(overlayPreviewTimer.current)
    }
    overlayPreviewTimer.current = window.setTimeout(() => {
      previewOverlayConfig({
        opacity: overlay.opacity,
        font_size: overlay.font_size,
        source_font_size: overlay.source_font_size,
        show_source: overlay.show_source,
      }).catch(error => console.debug('[settings.overlay-preview] failed', error))
    }, 40)
  }, [])

  useEffect(() => {
    Promise.all([getConfig(), getPresets(), getPrompts(), getAudioDevices(), getModelStatus(), getCapabilities()])
      .then(([loadedConfig, loadedPresets, loadedPrompts, loadedDevices, loadedModels, loadedCapabilities]) => {
        const configuredPrompt = loadedConfig.translation.prompt_preset
        const configuredPromptExists = Boolean(loadedPrompts.presets[configuredPrompt])
        const syncedPrompts = configuredPromptExists
          ? { ...loadedPrompts, active: configuredPrompt }
          : loadedPrompts
        const fallbackPrompt = syncedPrompts.presets[syncedPrompts.active]
        const initialDraft = !configuredPromptExists && fallbackPrompt
          ? {
            ...loadedConfig,
            language: fallbackPrompt.source_language || loadedConfig.language,
            translation: {
              ...loadedConfig.translation,
              prompt_preset: syncedPrompts.active,
              target_language: fallbackPrompt.target_language || loadedConfig.translation.target_language,
            },
          }
          : loadedConfig
        setConfig(loadedConfig)
        setDraft(initialDraft)
        setPresets(loadedPresets)
        setPrompts(syncedPrompts)
        setDevices(loadedDevices)
        setCapabilities(loadedCapabilities)
        setModelStatus(`STT ${loadedModels.stt.exists ? 'found' : 'missing'} · LLM ${loadedModels.translation.exists ? 'found' : 'missing'}`)
      })
      .catch(error => console.debug('[settings.load] failed', error))
  }, [])

  useEffect(() => {
    if (!config || !draft || configsEqual(config, draft)) return
    const snapshot = draft
    const timer = window.setTimeout(() => {
      saveConfig(snapshot)
        .then(saved => {
          setConfig(saved)
          setDraft(current => current && configsEqual(current, snapshot) ? saved : current)
        })
        .catch(error => {
          console.debug('[settings.autosave] failed', error)
        })
    }, 450)
    return () => window.clearTimeout(timer)
  }, [config, draft])

  useEffect(() => () => {
    if (overlayPreviewTimer.current !== null) {
      window.clearTimeout(overlayPreviewTimer.current)
    }
  }, [])

  if (!draft) return <section className="panel loading">Loading settings...</section>

  const update = (patch: Partial<AppConfig>) => {
    setDraft(prev => ({ ...prev!, ...patch }))
  }

  const apply = async (id: string) => {
    const result = await applyPreset(id)
    setConfig(result.config)
    setDraft(result.config)
    setPendingPreset(null)
  }

  const savePromptState = async (next: PromptState | null = prompts) => {
    if (!next) return
    const saved = await savePrompts(next)
    syncPromptState(saved)
  }

  const resetPromptState = async (id?: string) => {
    const saved = await resetPrompt(id)
    syncPromptState(saved)
  }

  const deletePromptState = async (id: string) => {
    const saved = await deletePrompt(id)
    syncPromptState(saved)
  }

  const knownDeviceNames = new Set(devices.map(device => device.name))
  const configuredDevice = draft.audio.device_name && !knownDeviceNames.has(draft.audio.device_name)
    ? [{ name: draft.audio.device_name, hostapi: 'Configured', recommended: false }]
    : []
  const audioDeviceOptions = [...configuredDevice, ...devices].map(device => ({
    label: device.name,
    value: device.name,
    badge: device.recommended ? 'Recommended · WASAPI' : device.hostapi,
    recommended: device.recommended,
  }))
  const activePrompt = prompts?.presets[prompts.active]
  const sourceLanguage = capabilityLanguageCode(activePrompt?.source_language || draft.language, capabilities)
  const targetLanguage = capabilityLanguageCode(activePrompt?.target_language || draft.translation.target_language, capabilities)
  const availableSttModels = capabilities?.stt_models.filter(model => (
    model.languages.includes(sourceLanguage)
  )) ?? []
  const defaultSttId = capabilities?.stt_defaults[sourceLanguage]
  const defaultStt = availableSttModels.find(model => model.value === defaultSttId) ?? availableSttModels[0]
  const sttModelIsKnown = draft.stt.sherpa_model_id === 'auto'
    || availableSttModels.some(model => model.value === draft.stt.sherpa_model_id)
  const standardSttModelOptions = availableSttModels.length <= 1
    ? [{ value: 'auto', label: defaultStt?.label ?? 'No compatible ASR model' }]
    : [
      { value: 'auto', label: `Auto (${defaultStt?.label ?? 'recommended'})` },
      ...availableSttModels.map(model => ({ value: model.value, label: model.label })),
    ]
  const sttModelOptions = sttModelIsKnown
    ? standardSttModelOptions
    : [{ value: draft.stt.sherpa_model_id, label: 'Custom', badge: draft.stt.sherpa_model_id }, ...standardSttModelOptions]
  const sttModelValue = sttModelIsKnown && availableSttModels.length <= 1
    ? 'auto'
    : draft.stt.sherpa_model_id
  const compatibleTranslationModels = capabilities?.translation_models.filter(model => (
    model.languages.includes(sourceLanguage) && model.languages.includes(targetLanguage)
  )) ?? []
  const translationModelIsKnown = compatibleTranslationModels.some(model => (
    sameModelFile(model.value, draft.translation.model_path)
  ))
  const standardTranslationModelOptions = compatibleTranslationModels.map(model => ({
    value: sameModelFile(model.value, draft.translation.model_path)
      ? draft.translation.model_path
      : model.value,
    label: model.label,
    recommended: model.recommended,
  }))
  const translationModelOptions = translationModelIsKnown
    ? standardTranslationModelOptions
    : [{
      value: draft.translation.model_path,
      label: 'Custom',
      badge: modelFileName(draft.translation.model_path),
    }, ...standardTranslationModelOptions]
  const standardSttDeviceOptions = capabilities?.stt_providers ?? [{ label: 'CPU (Recommended)', value: 'cpu' }]
  const sttDeviceOptions = withCustomOption(standardSttDeviceOptions, draft.stt.sherpa_onnx_provider)
  const translationDeviceOptions = withCustomOption(TRANSLATION_DEVICE_OPTIONS, draft.translation.device)
  const promptPresetOptions = prompts
    ? Object.entries(prompts.presets).map(([id, preset]) => ({
      value: id,
      label: promptPresetLabel(preset),
    }))
    : []

  function syncPromptState(next: PromptState) {
    setPrompts(next)
    const prompt = next.presets[next.active]
    if (!prompt) return
    setDraft(prev => {
      if (!prev) return prev
      const source = capabilityLanguageCode(prompt.source_language, capabilities)
      const target = capabilityLanguageCode(prompt.target_language, capabilities)
      const allowedStt = capabilities?.stt_models.filter(model => model.languages.includes(source)) ?? []
      const allowedTranslation = capabilities?.translation_models.filter(model => (
        model.languages.includes(source) && model.languages.includes(target)
      )) ?? []
      const selectedTranslation = allowedTranslation.some(model => sameModelFile(model.value, prev.translation.model_path))
        ? prev.translation.model_path
        : allowedTranslation[0]?.value ?? prev.translation.model_path
      return {
        ...prev,
        language: prompt.source_language || prev.language,
        stt: {
          ...prev.stt,
          sherpa_model_id: prev.stt.sherpa_model_id === 'auto'
            || allowedStt.some(model => model.value === prev.stt.sherpa_model_id)
            ? prev.stt.sherpa_model_id
            : 'auto',
        },
        translation: {
          ...prev.translation,
          prompt_preset: next.active,
          target_language: prompt.target_language || prev.translation.target_language,
          model_path: selectedTranslation,
        },
      }
    })
  }

  return (
    <section className="settings-layout">
      <div className="panel">
        <div className="section-head">
          <div>
            <h1>Settings</h1>
            <p>{modelStatus || 'Model status will appear after load'}</p>
          </div>
        </div>

        <div className="settings-sections">
          <section className="settings-section">
            <div className="settings-section-head">
              <span className="settings-section-icon">
                <SlidersHorizontal size={20} strokeWidth={2.4} />
              </span>
              <div>
                <h2>General settings</h2>
                <p>Languages, audio input and browser subtitles.</p>
              </div>
            </div>

            <div className="form-grid">
              {prompts && (
                <SelectField
                  label="Translation preset"
                  value={prompts.active}
                  options={promptPresetOptions}
                  searchable
                  searchPlaceholder="Search preset..."
                  className="span-3"
                  onChange={value => syncPromptState({ ...prompts, active: value })}
                />
              )}

              {/* Kept in config for future desktop capture modes. */}
              {/* <SelectField
                label="Audio source"
                value={draft.audio.capture_mode}
                options={[
                  { value: 'browser_tab', label: 'Browser tab (Recommended)' },
                  { value: 'input_device', label: 'Audio input device' },
                ]}
                onChange={value => update({
                  audio: { ...draft.audio, capture_mode: value },
                })}
              /> */}

              {draft.audio.capture_mode === 'input_device' && (
                <SelectField
                  label="Input device"
                  value={draft.audio.device_name}
                  options={audioDeviceOptions}
                  onChange={value => update({
                    audio: { ...draft.audio, device_name: value },
                  })}
                />
              )}

              <label className="field">
                <span>Subtitle font size</span>
                <input
                  type="number"
                  value={draft.overlay.font_size}
                  onChange={event => {
                    const fontSize = Number(event.target.value)
                    const overlay = {
                      ...draft.overlay,
                      font_size: fontSize,
                      source_font_size: Math.max(8, Math.round(fontSize * 15 / 23)),
                    }
                    update({ overlay })
                    pushOverlayPreview(overlay)
                  }}
                />
              </label>

              <OpacitySlider
                value={draft.overlay.opacity}
                onCommit={value => update({ overlay: { ...draft.overlay, opacity: value } })}
                onPreview={value => pushOverlayPreview({ ...draft.overlay, opacity: value })}
              />

              <div className="field check-card-field">
                <span>Subtitle context</span>
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={draft.overlay.show_source}
                    onChange={event => {
                      const overlay = { ...draft.overlay, show_source: event.target.checked }
                      update({ overlay })
                      pushOverlayPreview(overlay)
                    }}
                  />
                  <span>Show previous line</span>
                </label>
              </div>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-head">
              <span className="settings-section-icon">
                <Gauge size={20} strokeWidth={2.4} />
              </span>
              <div>
                <h2>Performance</h2>
                <p>VRAM presets, ASR model and translation runtime.</p>
              </div>
            </div>

            <div className="preset-grid">
              {presets.map(preset => (
                <button
                  key={preset.id}
                  className={`preset-card ${preset.custom ? 'custom' : ''}`}
                  onClick={() => setPendingPreset(preset)}
                >
                  <span>{preset.vram}</span>
                  <strong>{preset.name}</strong>
                  <small>{preset.quality} · {preset.speed}</small>
                </button>
              ))}
            </div>

            <div className="form-grid performance-fields">
              <SelectField
                label="Translation model"
                className="performance-half"
                value={draft.translation.model_path}
                options={translationModelOptions}
                onChange={value => update({
                  translation: { ...draft.translation, model_path: value },
                })}
              />

              <SelectField
                label="Translation device"
                className="performance-half"
                value={draft.translation.device}
                options={translationDeviceOptions}
                onChange={value => update({
                  translation: {
                    ...draft.translation,
                    device: value,
                    n_gpu_layers: value === 'gpu' && draft.translation.n_gpu_layers === 0
                      ? -1
                      : draft.translation.n_gpu_layers,
                  },
                })}
              />

              <SelectField
                label="ASR model"
                className="performance-third"
                value={sttModelValue}
                options={sttModelOptions}
                disabled={availableSttModels.length <= 1 && sttModelIsKnown}
                onChange={value => update({ stt: { ...draft.stt, sherpa_model_id: value } })}
              />

              <SelectField
                label="ASR device"
                className="performance-third"
                value={draft.stt.sherpa_onnx_provider}
                options={sttDeviceOptions}
                onChange={value => update({ stt: { ...draft.stt, sherpa_onnx_provider: value } })}
              />

              <SelectField
                label="Context subtitles"
                className="performance-third"
                value={String(draft.translation.context_subtitles)}
                options={CONTEXT_SUBTITLE_OPTIONS}
                onChange={value => {
                  const contextSubtitles = Number(value)
                  update({
                    translation: {
                      ...draft.translation,
                      context_subtitles: contextSubtitles,
                      n_ctx: contextWindowForSubtitles(contextSubtitles),
                    },
                  })
                }}
              />
            </div>
          </section>
        </div>
      </div>

      <PromptSettingsPanel
        prompts={prompts}
        onChange={syncPromptState}
        onSave={savePromptState}
        onReset={resetPromptState}
        onDelete={deletePromptState}
        capabilities={capabilities}
      />

      <div className="panel advanced-toggle">
        <label className="check-field">
          <input type="checkbox" checked={advanced} onChange={event => setAdvanced(event.target.checked)} />
          <span>Advanced</span>
        </label>
      </div>

      <AdvancedSettingsPanel
        config={draft}
        showEditor={advanced}
        onConfigChange={next => { setConfig(next); setDraft(next) }}
      />
      <PresetComparisonDialog
        open={Boolean(pendingPreset)}
        presetName={pendingPreset?.name ?? ''}
        changes={pendingPreset ? presetChanges(draft, pendingPreset.patch) : []}
        onConfirm={() => pendingPreset && apply(pendingPreset.id)}
        onCancel={() => setPendingPreset(null)}
      />
    </section>
  )
}

const OpacitySlider = memo(function OpacitySlider({
  value,
  onCommit,
  onPreview,
}: {
  value: number
  onCommit: (value: number) => void
  onPreview: (value: number) => void
}) {
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const commit = () => {
    if (Math.abs(localValue - value) > 0.0005) {
      onCommit(localValue)
    }
  }

  return (
    <label className="field range-field">
      <span>
        Subtitle opacity
        <b>{Math.round(localValue * 100)}%</b>
      </span>
      <input
        type="range"
        min="0.45"
        max="1"
        step="0.001"
        value={localValue}
        style={{ '--range-progress': `${((localValue - 0.45) / 0.55) * 100}%` } as CSSProperties}
        onChange={event => {
          const next = Number(event.target.value)
          setLocalValue(next)
          onPreview(next)
        }}
        onBlur={commit}
        onKeyUp={commit}
        onMouseUp={commit}
        onPointerUp={commit}
        onTouchEnd={commit}
      />
    </label>
  )
})

function presetChanges(config: AppConfig, patch: Partial<AppConfig>) {
  const changes: { label: string; current: string; next: string }[] = []
  const walk = (current: unknown, next: unknown, path: string[]) => {
    if (next && typeof next === 'object' && !Array.isArray(next)) {
      for (const [key, value] of Object.entries(next)) {
        walk(current && typeof current === 'object' ? (current as Record<string, unknown>)[key] : undefined, value, [...path, key])
      }
      return
    }
    if (JSON.stringify(current) !== JSON.stringify(next)) {
      changes.push({
        label: path.map(part => part.replaceAll('_', ' ')).join(' · '),
        current: String(current ?? 'not set'),
        next: String(next ?? 'not set'),
      })
    }
  }
  walk(config, patch, [])
  return changes
}
