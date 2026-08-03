import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createLogsSocket, pauseRuntime, restartRuntime, resumeRuntime, startRuntime, stopRuntime } from '../../api/client'
import type { RuntimeStatus } from '../../types'

const MAX_LOG_LINES = 500
const LOG_BOTTOM_EPSILON_PX = 28
const PHASE_LABELS: Record<string, string> = {
  checking_models: 'Checking models',
  initializing_stt: 'Initializing STT',
  initializing_llm: 'Initializing LLM',
  ready: 'Ready',
  transcribing: 'Transcribing',
  translating: 'Translating',
}

type Props = {
  runtime: RuntimeStatus
  onRuntimeChange: (status: RuntimeStatus) => void
}

export default function ControlPage({ runtime, onRuntimeChange }: Props) {
  const [logs, setLogs] = useState<string[]>([])
  const [logsPinned, setLogsPinned] = useState(true)
  const terminalRef = useRef<HTMLDivElement | null>(null)
  const logsPinnedRef = useRef(true)

  useEffect(() => {
    const ws = createLogsSocket(line => setLogs(prev => {
      const progressKey = line.startsWith('[download:') ? line.slice(0, line.indexOf(']') + 1) : ''
      if (progressKey && prev.at(-1)?.startsWith(progressKey)) {
        return [...prev.slice(0, -1), line]
      }
      return [...prev.slice(-(MAX_LOG_LINES - 1)), line]
    }))
    ws.onerror = error => console.debug('[logs.ws] error', error)
    return () => ws.close()
  }, [])

  useLayoutEffect(() => {
    if (!logsPinnedRef.current) return
    const terminal = terminalRef.current
    if (!terminal) return
    terminal.scrollTop = terminal.scrollHeight
  }, [logs])

  const isRunning = runtime.state === 'running' || runtime.state === 'starting' || runtime.state === 'paused'
  const canPause = runtime.state === 'running' || runtime.state === 'starting'
  const canStop = runtime.state === 'running' || runtime.state === 'starting' || runtime.state === 'paused' || runtime.state === 'error'
  const statusCopy = useMemo(() => {
    if (runtime.last_error) return runtime.last_error
    if (runtime.pid) return `Process ${runtime.pid} is attached`
    if (runtime.exit_code !== null) return `Last exit code ${runtime.exit_code}`
    return 'Ready to start realtime translation'
  }, [runtime])
  const updateLogPinState = () => {
    const terminal = terminalRef.current
    if (!terminal) return
    const distanceToBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight
    const nextPinned = distanceToBottom <= LOG_BOTTOM_EPSILON_PX
    logsPinnedRef.current = nextPinned
    setLogsPinned(nextPinned)
  }

  const jumpToLatest = () => {
    const terminal = terminalRef.current
    if (!terminal) return
    logsPinnedRef.current = true
    setLogsPinned(true)
    terminal.scrollTop = terminal.scrollHeight
  }

  const clearLogs = () => {
    logsPinnedRef.current = true
    setLogsPinned(true)
    setLogs([])
  }

  const call = async (action: () => Promise<{ runtime: RuntimeStatus }>) => {
    try {
      const result = await action()
      onRuntimeChange(result.runtime)
    } catch (error) {
      console.debug('[control.action] failed', error)
    }
  }

  return (
    <section className="control-layout">
      <div className="control-rail left" aria-hidden="true" />

      <div className="control-main panel">
        <div className="control-hero">
          <div>
            <span className="control-kicker">Runtime control</span>
            <h1>Realtime Translator</h1>
            <p>{statusCopy}</p>
          </div>
          <div className={`state-badge ${runtime.state}`}>{runtime.state}</div>
        </div>

        <div className="control-actions">
          <button className="btn primary start-button control-action-main" disabled={isRunning} onClick={() => call(startRuntime)}>
            Start
          </button>
          <button
            className="btn control-action"
            disabled={!canPause && runtime.state !== 'paused'}
            onClick={() => call(runtime.state === 'paused' ? resumeRuntime : pauseRuntime)}
          >
            {runtime.state === 'paused' ? 'Resume' : 'Pause'}
          </button>
          <button className="btn danger control-action" disabled={!canStop} onClick={() => call(stopRuntime)}>
            Stop
          </button>
          <button className="btn control-action" onClick={() => call(restartRuntime)}>
            Restart
          </button>
        </div>

        <div className="control-meta">
          <MetaItem label="PID" value={runtime.pid ? String(runtime.pid) : 'not running'} />
          <MetaItem label="Paused" value={runtime.paused ? 'yes' : 'no'} />
          <MetaItem label="Last exit" value={runtime.exit_code === null ? 'none' : String(runtime.exit_code)} />
        </div>
      </div>

      <aside className="log-panel panel">
        <div className="section-head compact log-head">
          <h2>Logs</h2>
          <div className="log-head-actions">
            <span className={`runtime-phase ${runtime.phase || runtime.state}`} aria-live="polite">
              <span className="runtime-phase-dot" aria-hidden="true" />
              {PHASE_LABELS[runtime.phase] || runtime.phase || runtime.state}
            </span>
            {!logsPinned && (
              <button className="btn small" type="button" onClick={jumpToLatest}>
                Jump to latest
              </button>
            )}
            <button className="btn small log-clear" type="button" onClick={clearLogs} disabled={!logs.length}>
              Clear
            </button>
          </div>
        </div>
        <div ref={terminalRef} className="terminal" role="log" onScroll={updateLogPinState}>
          {logs.length
            ? logs.map((line, index) => <LogLine key={`${index}-${line}`} line={line} />)
            : <div className="log-line muted">Logs will stream here after start.</div>}
        </div>
      </aside>

      <div className="control-rail right" aria-hidden="true" />
    </section>
  )
}

function LogLine({ line }: { line: string }) {
  const warning = /(?:^|\s)(?:WARNING)(?:\s|$)/.test(line)
  const tagged = line.match(/^(\[(?:Source|Transl)\])(.*)$/)
  return (
    <div className={`log-line ${warning ? 'warning' : ''}`}>
      {tagged ? (
        <>
          <span className={`log-tag ${tagged[1] === '[Source]' ? 'source' : 'translation'}`}>{tagged[1]}</span>
          <span>{tagged[2]}</span>
        </>
      ) : line}
    </div>
  )
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="control-meta-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
