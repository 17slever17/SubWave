export type RuntimeState = 'idle' | 'downloading' | 'starting' | 'running' | 'paused' | 'stopping' | 'error'
export type RuntimePhase = RuntimeState | 'checking_models' | 'initializing_stt' | 'initializing_llm' | 'ready' | 'transcribing' | 'translating'

export type RuntimeStatus = {
  state: RuntimeState
  phase: RuntimePhase
  pid: number | null
  started_at: number | null
  stopped_at: number | null
  exit_code: number | null
  last_error: string
  paused: boolean
}

export type AppConfig = {
  _hash?: string
  language: string
  log_level: string
  stt: {
    sherpa_model_id: string
    sherpa_onnx_provider: string
    process_interval_s: number
    max_phrase_s: number
    [key: string]: unknown
  }
  translation: {
    model_path: string
    device: string
    n_gpu_layers: number
    mtp_enabled: boolean
    mtp_model_path: string
    mtp_n: number
    llama_server_path: string
    n_ctx: number
    n_batch: number
    context_subtitles: number
    target_language: string
    prompt_preset: string
    [key: string]: unknown
  }
  audio: {
    capture_mode: string
    device_name: string
    input_gain: number
    dfn2: Record<string, unknown>
    [key: string]: unknown
  }
  overlay: {
    enabled: boolean
    mode: string
    opacity: number
    font_size: number
    source_font_size: number
    show_source: boolean
    [key: string]: unknown
  }
  [key: string]: unknown
}

export type Preset = {
  id: string
  name: string
  vram: string
  quality: string
  speed: string
  description: string
  patch: Partial<AppConfig>
  custom?: boolean
}

export type AudioDevice = {
  name: string
  hostapi: string
  recommended: boolean
}

export type PromptPreset = {
  name: string
  source_language: string
  target_language: string
  template: string
}

export type PromptState = {
  active: string
  presets: Record<string, PromptPreset>
  factory_ids?: string[]
}

export type HealthCheck = {
  name: string
  ok: boolean
  detail: string
}

export type ModelCapability = {
  value: string
  label: string
  languages: string[]
  installed?: boolean
  family?: string
  recommended?: boolean
}

export type Capabilities = {
  languages: { value: string; label: string }[]
  source_languages: string[]
  target_languages: string[]
  stt_models: ModelCapability[]
  stt_defaults: Record<string, string>
  stt_providers: { value: string; label: string }[]
  translation_models: ModelCapability[]
}
