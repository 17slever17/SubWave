import { useEffect, useState } from 'react'
import ControlPage from './components/control/ControlPage'
import SettingsPage from './components/settings/SettingsPage'
import { createStatusSocket, getStatus } from './api/client'
import type { RuntimeStatus } from './types'

type TabId = 'control' | 'settings'

const IDLE_STATUS: RuntimeStatus = {
  state: 'idle',
  phase: 'idle',
  pid: null,
  started_at: null,
  stopped_at: null,
  exit_code: null,
  last_error: '',
  paused: false,
}

function tabFromPath(pathname: string): TabId {
  return pathname.replace(/\/+$/, '') === '/settings' ? 'settings' : 'control'
}

function normalizeRoute(): TabId {
  const tab = tabFromPath(window.location.pathname)
  if (window.location.pathname === '/' || !['/control', '/settings'].includes(window.location.pathname.replace(/\/+$/, ''))) {
    window.history.replaceState(null, '', `/${tab}`)
  }
  return tab
}

export default function App() {
  const [tab, setTab] = useState<TabId>(() => normalizeRoute())
  const [runtime, setRuntime] = useState<RuntimeStatus>(IDLE_STATUS)

  useEffect(() => {
    const onPopState = () => setTab(normalizeRoute())
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => {
    getStatus()
      .then(payload => setRuntime(payload.runtime))
      .catch(error => console.debug('[app.status] initial load failed', error))
    const ws = createStatusSocket(setRuntime)
    ws.onerror = error => console.debug('[app.status] websocket error', error)
    return () => ws.close()
  }, [])

  const navigate = (next: TabId) => {
    const path = `/${next}`
    if (window.location.pathname !== path) {
      window.history.pushState(null, '', path)
    }
    setTab(next)
  }

  return (
    <div className="app-shell">
      <div className="ambient-grid" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-kicker">local stream</div>
          <div className="brand-title">Realtime Translator</div>
        </div>

        <nav className="tabs" aria-label="Main navigation">
          <button className={`tab ${tab === 'control' ? 'active' : ''}`} onClick={() => navigate('control')}>
            Control
          </button>
          <button className={`tab ${tab === 'settings' ? 'active' : ''}`} onClick={() => navigate('settings')}>
            Settings
          </button>
        </nav>

        <div className={`runtime-pill ${runtime.state}`}>
          <span className="status-light" />
          <span>{runtime.state}</span>
        </div>
      </header>

      <main className="content">
        {tab === 'control' ? <ControlPage runtime={runtime} onRuntimeChange={setRuntime} /> : <SettingsPage />}
      </main>
    </div>
  )
}
