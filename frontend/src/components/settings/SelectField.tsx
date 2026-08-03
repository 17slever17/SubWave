import { useEffect, useMemo, useRef, useState } from 'react'

export type SelectOption = {
  label: string
  value: string
  badge?: string
  recommended?: boolean
}

type Props = {
  label: string
  value: string
  options: SelectOption[]
  onChange: (value: string) => void
  searchable?: boolean
  searchPlaceholder?: string
  className?: string
  disabled?: boolean
}

export default function SelectField({ label, value, options, onChange, searchable = false, searchPlaceholder = 'Search...', className = '', disabled = false }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const searchInputRef = useRef<HTMLInputElement | null>(null)
  const selected = useMemo(() => options.find(option => option.value === value) ?? options[0], [options, value])
  const filteredOptions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return options
    return options.filter(option => (
      option.label.toLowerCase().includes(normalizedQuery)
      || option.value.toLowerCase().includes(normalizedQuery)
    ))
  }, [options, query])

  useEffect(() => {
    if (!open) {
      setQuery('')
      return
    }
    if (searchable) {
      window.setTimeout(() => searchInputRef.current?.focus(), 0)
    }
  }, [open, searchable])

  return (
    <label className={`field select-field ${disabled ? 'disabled' : ''} ${className}`}>
      <span>{label}</span>
      <div
        className={`select-shell ${open ? 'open' : ''}`}
        onBlur={event => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setOpen(false)
          }
        }}
      >
        {open && searchable ? (
          <>
            <input
              ref={searchInputRef}
              className="select-button select-button-search"
              aria-label={`${label} search`}
              value={query}
              placeholder={selected?.label ?? searchPlaceholder}
              onChange={event => setQuery(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Escape') setOpen(false)
              }}
            />
            <i className="select-arrow" aria-hidden="true" />
          </>
        ) : (
          <button
            className="select-button"
            type="button"
            title={selected?.label}
            aria-label={`${label}: ${selected?.label ?? 'Select'}`}
            disabled={disabled}
            onClick={() => setOpen(prev => !prev)}
          >
            <span>{selected?.label ?? 'Select...'}</span>
            {selected?.badge && <b className={`select-value-badge ${selected.recommended ? 'recommended' : ''}`}>{selected.badge}</b>}
            <i className="select-arrow" aria-hidden="true" />
          </button>
        )}

        {open && (
          <div className="select-menu" role="listbox">
            {filteredOptions.map(option => (
              <button
                key={option.value}
                className={`select-option ${option.value === value ? 'active' : ''} ${option.recommended ? 'recommended' : ''}`}
                type="button"
                title={option.label}
                role="option"
                aria-selected={option.value === value}
                onMouseDown={event => event.preventDefault()}
                onClick={() => {
                  onChange(option.value)
                  setOpen(false)
                }}
              >
                <span>{option.label}</span>
                {option.badge && <b>{option.badge}</b>}
              </button>
            ))}
            {!filteredOptions.length && <div className="select-empty">Nothing found</div>}
          </div>
        )}
      </div>
    </label>
  )
}
