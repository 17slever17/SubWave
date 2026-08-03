import axios from 'axios'
import type { AppConfig, AudioDevice, Capabilities, HealthCheck, Preset, PromptState, RuntimeStatus } from '../types'

const api = axios.create({ baseURL: '' })

export const getStatus = () => api.get<{ runtime: RuntimeStatus; config_hash: string }>('/api/status').then(r => r.data)
export const startRuntime = () => api.post<{ runtime: RuntimeStatus }>('/api/control/start').then(r => r.data)
export const pauseRuntime = () => api.post<{ runtime: RuntimeStatus }>('/api/control/pause').then(r => r.data)
export const resumeRuntime = () => api.post<{ runtime: RuntimeStatus }>('/api/control/resume').then(r => r.data)
export const stopRuntime = () => api.post<{ runtime: RuntimeStatus }>('/api/control/stop').then(r => r.data)
export const restartRuntime = () => api.post<{ runtime: RuntimeStatus }>('/api/control/restart').then(r => r.data)

export const getConfig = () => api.get<AppConfig>('/api/config').then(r => r.data)
export const saveConfig = (config: AppConfig) => api.post<{ config: AppConfig }>('/api/config', config).then(r => r.data.config)
export const previewOverlayConfig = (
  overlay: Pick<AppConfig['overlay'], 'opacity' | 'font_size' | 'source_font_size' | 'show_source'>,
) => (
  api.post<{ ok: boolean }>('/api/config/overlay-preview', overlay).then(r => r.data)
)
export const resetConfigSection = (section: string) => api.post<{ config: AppConfig }>('/api/config/reset-section', { section }).then(r => r.data.config)

export const getPresets = () => api.get<{ presets: Preset[] }>('/api/presets').then(r => r.data.presets)
export const applyPreset = (id: string) => api.post<{ preset: Preset; config: AppConfig }>('/api/presets/apply', { id }).then(r => r.data)

export const getPrompts = () => api.get<PromptState>('/api/prompts').then(r => r.data)
export const savePrompts = (state: PromptState) => api.post<PromptState>('/api/prompts', state).then(r => r.data)
export const resetPrompt = (id?: string) => api.post<PromptState>('/api/prompts/reset', id ? { id } : {}).then(r => r.data)
export const deletePrompt = (id: string) => api.post<PromptState>('/api/prompts/delete', { id }).then(r => r.data)
export const getCapabilities = () => api.get<Capabilities>('/api/capabilities').then(r => r.data)

export const getAudioDevices = () => api.get<{ devices: (AudioDevice | string)[] }>('/api/audio/devices').then(r => {
  const normalized = r.data.devices.map(device => typeof device === 'string'
    ? {
      name: device,
      hostapi: 'Audio input',
      recommended: device === 'CABLE Output (VB-Audio Virtual Cable)',
    }
    : device)
  const unique = new Map<string, AudioDevice>()
  for (const device of normalized) {
    const current = unique.get(device.name)
    if (!current || device.recommended) unique.set(device.name, device)
  }
  return [...unique.values()].sort((left, right) => Number(right.recommended) - Number(left.recommended) || left.name.localeCompare(right.name))
})
export const getHealth = () => api.get<{ ok: boolean; checks: HealthCheck[] }>('/api/health').then(r => r.data)
export const getModelStatus = () => api.get<{ stt: { path: string; exists: boolean }; translation: { path: string; exists: boolean } }>('/api/models/status').then(r => r.data)

function wsBase(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = import.meta.env.DEV && import.meta.env.VITE_REALTIME_BACKEND_PORT
    ? `${window.location.hostname}:${import.meta.env.VITE_REALTIME_BACKEND_PORT}`
    : window.location.host
  return `${proto}//${host}`
}

export function createLogsSocket(onMessage: (line: string) => void): WebSocket {
  const ws = new WebSocket(`${wsBase()}/ws/logs`)
  ws.onmessage = event => {
    try {
      const payload = JSON.parse(event.data)
      if (payload?.text) onMessage(String(payload.text))
    } catch {
      onMessage(String(event.data))
    }
  }
  return ws
}

export function createStatusSocket(onMessage: (status: RuntimeStatus) => void): WebSocket {
  const ws = new WebSocket(`${wsBase()}/ws/status`)
  ws.onmessage = event => {
    try {
      const payload = JSON.parse(event.data)
      if (payload?.runtime) onMessage(payload.runtime)
    } catch {
      console.debug('[status.ws] ignored malformed payload', event.data)
    }
  }
  return ws
}
