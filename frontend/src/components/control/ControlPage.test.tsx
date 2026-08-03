import { act, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { RuntimeStatus } from '../../types'
import ControlPage from './ControlPage'

const socket = vi.hoisted(() => ({
  onLog: null as ((line: string) => void) | null,
}))

vi.mock('../../api/client', () => ({
  createLogsSocket: vi.fn((onMessage: (line: string) => void) => {
    socket.onLog = onMessage
    return { close: vi.fn() }
  }),
  pauseRuntime: vi.fn(),
  restartRuntime: vi.fn(),
  resumeRuntime: vi.fn(),
  startRuntime: vi.fn(),
  stopRuntime: vi.fn(),
}))

const runtime: RuntimeStatus = {
  state: 'running',
  phase: 'ready',
  pid: 42,
  started_at: 1,
  stopped_at: null,
  exit_code: null,
  last_error: '',
  paused: false,
}

describe('ControlPage logs', () => {
  it('shows the runtime phase and clears only visible log lines', () => {
    render(<ControlPage runtime={runtime} onRuntimeChange={vi.fn()} />)

    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(screen.queryByText(/lines$/)).not.toBeInTheDocument()

    act(() => socket.onLog?.('A runtime log line'))
    expect(screen.getByText('A runtime log line')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(screen.queryByText('A runtime log line')).not.toBeInTheDocument()
    expect(screen.getByText('Ready')).toBeInTheDocument()
  })

  it('updates one download progress line instead of growing the log', () => {
    render(<ControlPage runtime={runtime} onRuntimeChange={vi.fn()} />)

    act(() => socket.onLog?.('[download:STT model] 10.0% | 10/100 MiB | 2.0 MiB/s | ETA 00:45'))
    act(() => socket.onLog?.('[download:STT model] 20.0% | 20/100 MiB | 3.0 MiB/s | ETA 00:27'))

    expect(screen.queryByText(/10\.0%/)).not.toBeInTheDocument()
    expect(screen.getByText(/20\.0%/)).toBeInTheDocument()
  })
})
