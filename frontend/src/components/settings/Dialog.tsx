import { useEffect, useState } from 'react'

type ConfirmDialogProps = {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}

type TextInputDialogProps = {
  open: boolean
  title: string
  label: string
  initialValue: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: (value: string) => void
  onCancel: () => void
}

type ChangePreview = { label: string; current: string; next: string }

type PresetComparisonDialogProps = {
  open: boolean
  presetName: string
  changes: ChangePreview[]
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!open) return null

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title" onMouseDown={event => event.stopPropagation()}>
        <h3 id="confirm-dialog-title">{title}</h3>
        <p>{message}</p>
        <div className="dialog-actions">
          <button className="btn" onClick={onCancel}>{cancelLabel}</button>
          <button className={`btn ${danger ? 'danger' : 'primary'}`} onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}

export function TextInputDialog({
  open,
  title,
  label,
  initialValue,
  confirmLabel = 'Create',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
}: TextInputDialogProps) {
  const [value, setValue] = useState(initialValue)

  useEffect(() => {
    if (open) setValue(initialValue)
  }, [initialValue, open])

  if (!open) return null

  const submit = () => {
    const trimmed = value.trim()
    if (trimmed) onConfirm(trimmed)
  }

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="input-dialog-title" onMouseDown={event => event.stopPropagation()}>
        <h3 id="input-dialog-title">{title}</h3>
        <label className="field">
          <span>{label}</span>
          <input value={value} autoFocus onChange={event => setValue(event.target.value)} onKeyDown={event => {
            if (event.key === 'Enter') submit()
            if (event.key === 'Escape') onCancel()
          }} />
        </label>
        <div className="dialog-actions">
          <button className="btn" onClick={onCancel}>{cancelLabel}</button>
          <button className="btn primary" onClick={submit} disabled={!value.trim()}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}

export function PresetComparisonDialog({ open, presetName, changes, onConfirm, onCancel }: PresetComparisonDialogProps) {
  if (!open) return null
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div className="dialog-card preset-comparison" role="dialog" aria-modal="true" onMouseDown={event => event.stopPropagation()}>
        <h3>Apply {presetName}?</h3>
        <p>Only the settings below will change.</p>
        <div className="change-list">
          {changes.length ? changes.map(change => (
            <div className={`change-row ${change.current.length + change.next.length > 40 ? 'long' : ''}`} key={change.label}>
              <strong>{change.label}</strong>
              <span className="change-current">{change.current}</span>
              <i aria-hidden="true">→</i>
              <span className="change-next">{change.next}</span>
            </div>
          )) : <div className="change-empty">This preset already matches the current settings.</div>}
        </div>
        <div className="dialog-actions">
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn primary" onClick={onConfirm}>Apply</button>
        </div>
      </div>
    </div>
  )
}
