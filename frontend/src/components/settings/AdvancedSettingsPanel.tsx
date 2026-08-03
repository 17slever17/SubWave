import { useEffect, useMemo, useState } from 'react'
import { getHealth, resetConfigSection, saveConfig } from '../../api/client'
import type { AppConfig, HealthCheck } from '../../types'
import { ConfirmDialog } from './Dialog'

type Props = {
  config: AppConfig
  showEditor: boolean
  onConfigChange: (config: AppConfig) => void
}

const SECTIONS = [
  { id: 'audio', label: 'Audio' },
  { id: 'stt', label: 'STT' },
  { id: 'translation', label: 'Translation' },
  { id: 'overlay', label: 'Overlay' },
  { id: 'language', label: 'Language' },
  { id: 'log_level', label: 'Log level' },
] as const

type SectionId = (typeof SECTIONS)[number]['id']
type ConfigValue = null | boolean | number | string | ConfigValue[] | { [key: string]: ConfigValue }

export default function AdvancedSettingsPanel({ config, showEditor, onConfigChange }: Props) {
  const [activeSection, setActiveSection] = useState<SectionId>('audio')
  const [sectionDraft, setSectionDraft] = useState<ConfigValue>(config.audio as ConfigValue)
  const [health, setHealth] = useState<HealthCheck[]>([])
  const [healthRunning, setHealthRunning] = useState(false)
  const [healthStatus, setHealthStatus] = useState('')
  const [message, setMessage] = useState('')
  const [resetDialogOpen, setResetDialogOpen] = useState(false)

  const activeLabel = useMemo(
    () => SECTIONS.find(section => section.id === activeSection)?.label ?? activeSection,
    [activeSection],
  )

  useEffect(() => {
    setSectionDraft(toConfigValue(config[activeSection]))
  }, [activeSection, config])

  useEffect(() => {
    setMessage('')
  }, [activeSection])

  const saveSection = async () => {
    try {
      const next = { ...config, [activeSection]: sectionDraft } as AppConfig
      const saved = await saveConfig(next)
      onConfigChange(saved)
      setMessage(`Saved ${activeLabel}`)
    } catch (error) {
      console.debug('[settings.advanced.save] failed', error)
      setMessage(`Could not save ${activeLabel}`)
    }
  }

  const runHealth = async () => {
    setHealthRunning(true)
    setHealthStatus('Running diagnostics...')
    try {
      const result = await getHealth()
      setHealth(result.checks)
      setHealthStatus(result.ok ? 'All checks passed' : 'Some checks need attention')
    } catch (error) {
      console.debug('[settings.health] failed', error)
      setHealthStatus('Health check failed')
    } finally {
      setHealthRunning(false)
    }
  }

  const resetSection = async () => {
    const next = await resetConfigSection(activeSection)
    onConfigChange(next)
    setResetDialogOpen(false)
    setMessage(`Reset ${activeLabel}`)
  }

  const updateValue = (path: Array<string | number>, value: ConfigValue) => {
    setSectionDraft(prev => setAtPath(prev, path, value))
  }

  return (
    <div className="advanced-stack">
      {showEditor && <div className="panel subpanel advanced-config-panel">
        <div className="section-head compact advanced-head">
          <div>
            <h2>Advanced config</h2>
            <p>Values only. Names and structure are locked.</p>
          </div>
          <span>{message}</span>
        </div>

        <div className="advanced-tabs" role="tablist" aria-label="Advanced config sections">
          {SECTIONS.map(section => (
            <button
              key={section.id}
              className={`advanced-tab ${activeSection === section.id ? 'active' : ''}`}
              type="button"
              role="tab"
              aria-selected={activeSection === section.id}
              onClick={() => setActiveSection(section.id)}
            >
              {section.label}
            </button>
          ))}
        </div>

        <div className="config-editor">
          <ConfigEditor value={sectionDraft} path={[]} label={activeLabel} onChange={updateValue} />
        </div>

        <div className="section-reset-row">
          <button className="btn primary" onClick={saveSection}>Save {activeLabel}</button>
          <button className="btn danger" onClick={() => setResetDialogOpen(true)}>Reset {activeLabel}</button>
        </div>
      </div>}

      <div className="panel subpanel">
        <div className="section-head compact health-head">
          <div>
            <h2>Health</h2>
            <p>{healthStatus || 'Run diagnostics when you want to verify local dependencies.'}</p>
          </div>
          <button className="btn small" disabled={healthRunning} onClick={runHealth}>
            {healthRunning ? 'Running...' : 'Run'}
          </button>
        </div>
        <div className="health-list">
          {health.length ? health.map(item => (
            <div key={item.name} className={`health-row ${item.ok ? 'ok' : 'bad'}`}>
              <strong>{item.name}</strong>
              <span>{item.detail || (item.ok ? 'ok' : 'failed')}</span>
            </div>
          )) : <div className="health-empty">Diagnostics have not been run yet.</div>}
        </div>
      </div>

      <ConfirmDialog
        open={resetDialogOpen}
        title={`Reset ${activeLabel}?`}
        message={`This will restore ${activeLabel} from config.reset.yaml and overwrite the current config.yaml section. Your current edits in this section will be lost.`}
        confirmLabel="Reset"
        danger
        onConfirm={resetSection}
        onCancel={() => setResetDialogOpen(false)}
      />
    </div>
  )
}

function ConfigEditor({
  value,
  path,
  label,
  onChange,
}: {
  value: ConfigValue
  path: Array<string | number>
  label: string
  onChange: (path: Array<string | number>, value: ConfigValue) => void
}) {
  if (Array.isArray(value)) {
    return (
      <div className="config-group">
        <div className="config-group-title">{label}</div>
        {value.length ? value.map((item, index) => (
          <ConfigEditor key={index} value={item} path={[...path, index]} label={`[${index}]`} onChange={onChange} />
        )) : <div className="config-empty">Empty list</div>}
      </div>
    )
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value)
    return (
      <div className="config-group">
        <div className="config-group-title">{label}</div>
        {entries.map(([key, item]) => (
          <ConfigEditor key={key} value={item} path={[...path, key]} label={key} onChange={onChange} />
        ))}
      </div>
    )
  }

  return (
    <label className="config-row">
      <span>{label}</span>
      <ConfigScalarInput value={value} onChange={next => onChange(path, next)} />
    </label>
  )
}

function ConfigScalarInput({ value, onChange }: { value: ConfigValue; onChange: (value: ConfigValue) => void }) {
  if (typeof value === 'boolean') {
    return (
      <label className="check-field config-check">
        <input type="checkbox" checked={value} onChange={event => onChange(event.target.checked)} />
        <span>{value ? 'Enabled' : 'Disabled'}</span>
      </label>
    )
  }

  if (typeof value === 'number') {
    return (
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        onChange={event => onChange(Number(event.target.value))}
      />
    )
  }

  return (
    <input
      value={value === null ? 'null' : String(value)}
      onChange={event => onChange(parseLooseScalar(event.target.value, value))}
    />
  )
}

function parseLooseScalar(raw: string, previous: ConfigValue): ConfigValue {
  if (previous === null) {
    const trimmed = raw.trim()
    if (trimmed === '' || trimmed.toLowerCase() === 'null') return null
    if (trimmed.toLowerCase() === 'true') return true
    if (trimmed.toLowerCase() === 'false') return false
    const numeric = Number(trimmed)
    if (trimmed && Number.isFinite(numeric)) return numeric
  }
  return raw
}

function setAtPath(root: ConfigValue, path: Array<string | number>, value: ConfigValue): ConfigValue {
  if (!path.length) return value
  const [head, ...tail] = path
  if (Array.isArray(root)) {
    return root.map((item, index) => index === head ? setAtPath(item, tail, value) : item)
  }
  if (isPlainObject(root)) {
    return {
      ...root,
      [head]: setAtPath(root[String(head)], tail, value),
    }
  }
  return root
}

function isPlainObject(value: ConfigValue): value is { [key: string]: ConfigValue } {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function toConfigValue(value: unknown): ConfigValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(toConfigValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toConfigValue(item)]))
  }
  return String(value ?? '')
}
