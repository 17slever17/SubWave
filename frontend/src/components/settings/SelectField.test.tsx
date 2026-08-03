import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import SelectField from './SelectField'

describe('SelectField', () => {
  it('filters searchable options in the same field and commits selection', () => {
    const onChange = vi.fn()
    render(
      <SelectField
        label="Translation preset"
        value="en_to_ru"
        searchable
        options={[
          { value: 'en_to_ru', label: 'English to Russian' },
          { value: 'ja_to_ru', label: 'Japanese to Russian' },
        ]}
        onChange={onChange}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /Translation preset: English to Russian/i }))
    const search = screen.getByRole('textbox', { name: 'Translation preset search' })
    fireEvent.change(search, { target: { value: 'japan' } })

    expect(screen.queryByRole('option', { name: 'English to Russian' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('option', { name: 'Japanese to Russian' }))
    expect(onChange).toHaveBeenCalledWith('ja_to_ru')
  })

  it('does not open or change a disabled field', () => {
    const onChange = vi.fn()
    render(
      <SelectField
        label="ASR model"
        value="auto"
        disabled
        options={[{ value: 'auto', label: 'Auto English' }]}
        onChange={onChange}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'ASR model: Auto English' }))
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })
})
