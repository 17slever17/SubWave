import os
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class STTConfig:
    sherpa_model_id: str = "auto"
    chunk_size_ms: int = 60
    process_interval_s: float = 2.0
    window_slack_s: float = 1.0
    process_batch_ms: int = 120
    window_overlap_s: float = 0.0
    end_vad_enabled: bool = True
    end_vad_backend: str = "firered"
    firered_vad_model_path: str = "models/FireRedVAD/Stream-VAD"
    firered_vad_threshold: float = 0.4
    firered_vad_min_speech_ms: int = 80
    firered_vad_rms_confirmation: bool = True
    ten_vad_model_path: str = "models/ten-vad.int8.onnx"
    ten_vad_threshold: float = 0.20
    ten_vad_min_speech_ms: int = 100
    end_vad_silence_ms: int = 230
    long_phrase_after_s: float = 6.0
    long_phrase_end_vad_silence_ms: int = 150
    max_phrase_s: float = 10.0
    forced_split_search_s: float = 3.0
    end_vad_rms_threshold: float = 0.010
    end_vad_frame_ms: int = 25
    end_vad_max_active_ratio: float = 0.15
    end_vad_recent_ms: int = 75
    speech_activity_rms_threshold: float = 0.018
    speech_activity_min_ratio: float = 0.12
    sherpa_onnx_provider: str = "cpu"
    sherpa_onnx_num_threads: int = 4
    sherpa_blank_penalty: Optional[float] = None
    blocked_phrases: list[str] = field(default_factory=lambda: ["Дима Торжок"])
    duplicate_similarity_threshold: float = 0.90
    max_consecutive_duplicate_utterances: int = 1
    max_consecutive_decode_failures: int = 3
    debug_audio_enabled: bool = False
    debug_audio_max_files: int = 100

@dataclass
class TranslationConfig:
    model_path: str = "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf"
    device: str = "gpu"
    n_gpu_layers: int = -1
    mtp_enabled: bool = True
    mtp_model_path: str = "auto"
    mtp_n: int = 1
    llama_server_path: str = "bin/llama.cpp/llama-server.exe"
    n_ctx: int = 1024
    n_batch: int = 256
    context_subtitles: int = 3
    target_language: str = "English"
    prompt_preset: str = "ja_to_en"
    max_tokens: int = 128
    repeat_penalty: float = 1.08
    duplicate_similarity_threshold: float = 0.90
    max_output_length_ratio: float = 5.0
    max_output_extra_chars: int = 80
    min_output_length_limit: int = 120
    max_output_chars: int = 500
    context_leak_similarity_threshold: float = 0.68
    context_leak_source_similarity_limit: float = 0.55
    context_max_age_s: float = 15.0
    realtime_queue_max_utterances: int = 2
    realtime_max_utterance_age_s: float = 3.0
    realtime_max_translation_latency_s: float = 3.0
    subtitle_duplicate_similarity_threshold: float = 0.82
    subtitle_duplicate_min_chars: int = 12
    subtitle_source_repeat_similarity_threshold: float = 0.78
    subtitle_cjk_source_repeat_similarity_threshold: float = 0.90

@dataclass
class Dfn2Config:
    enabled: bool = True
    input_sample_rate: int = 48000
    block_ms: int = 40
    vad_atten_lim_db: Optional[float] = 20.0
    stt_atten_lim_db: Optional[float] = 20.0
    vad_context_s: float = 3.0
    post_filter: bool = False
    device: str = "cpu"
    wet_mix: float = 1.0
    model: str = "DeepFilterNet2"
    log_level: str = "ERROR"


@dataclass
class FastEnhancerConfig:
    enabled: bool = True
    model_path: str = "models/fastenhancer/fastenhancer_b_48khz.onnx"
    input_sample_rate: int = 48000
    n_fft: int = 1024
    hop_samples: int = 512
    num_threads: int = 4
    wet_mix: float = 0.5
    wet_mix_by_language: dict[str, float] = field(
        default_factory=lambda: {"ja": 0.45}
    )
    languages: list[str] = field(
        default_factory=lambda: ["en", "ja", "ko", "es"]
    )


@dataclass
class AudioConfig:
    capture_mode: str = "browser_tab"
    device_name: str = "CABLE Output"
    fallback_mic_id: Optional[int] = None
    browser_audio_host: str = "127.0.0.1"
    browser_audio_port: int = 8766
    sampling_rate: int = 16000
    input_gain: float = 1.0
    buffer_max_chunks: int = 1024
    drop_oldest_on_overflow: bool = True
    max_pcm_backlog_s: float = 6.0
    dfn2: Dfn2Config = field(default_factory=Dfn2Config)
    fastenhancer: FastEnhancerConfig = field(
        default_factory=FastEnhancerConfig
    )


@dataclass
class OverlayConfig:
    enabled: bool = True
    mode: str = "browser"
    topmost: bool = True
    browser_bridge_enabled: bool = True
    browser_bridge_host: str = "127.0.0.1"
    browser_bridge_port: int = 8765
    browser_bridge_timeout_s: float = 2.0
    browser_targets: list[str] = field(default_factory=lambda: ["all"])
    title_keywords: list[str] = field(default_factory=lambda: ["twitch", "youtube"])
    opacity: float = 0.82
    width: int = 860
    min_width: int = 420
    max_width: int = 1200
    height: int = 124
    min_height: int = 88
    max_height: int = 220
    bottom_offset: int = 80
    font_size: int = 23
    source_font_size: int = 15
    show_source: bool = True
    corner_radius: int = 28
    background: str = "#101010"
    foreground: str = "#F5F5F5"
    source_foreground: str = "#B8B8B8"

@dataclass
class Config:
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    language: str = "ja"
    log_level: str = "INFO"
    translation_log_max_bytes: int = 10_000_000

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def context_window_for_subtitles(value: int) -> int:
    count = min(5, max(0, int(value)))
    return 544 + (count * 160)


def load_config(config_path: str | None = None) -> Config:
    config = Config()
    logger = logging.getLogger(__name__)
    resolved_path = Path(
        config_path
        or os.environ.get("REALTIME_TRANSLATOR_CONFIG")
        or DEFAULT_CONFIG_PATH
    )
    
    if resolved_path.exists():
        try:
            with resolved_path.open('r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                
            if 'stt' in yaml_data:
                for k, v in yaml_data['stt'].items():
                    setattr(config.stt, k, v)
            if 'translation' in yaml_data:
                for k, v in yaml_data['translation'].items():
                    setattr(config.translation, k, v)
            if 'audio' in yaml_data:
                for k, v in yaml_data['audio'].items():
                    if k == 'dfn2' and isinstance(v, dict):
                        for dk, dv in v.items():
                            setattr(config.audio.dfn2, dk, dv)
                    elif k == 'fastenhancer' and isinstance(v, dict):
                        for fk, fv in v.items():
                            setattr(config.audio.fastenhancer, fk, fv)
                    else:
                        setattr(config.audio, k, v)
            if 'overlay' in yaml_data:
                for k, v in yaml_data['overlay'].items():
                    setattr(config.overlay, k, v)
            if 'language' in yaml_data:
                config.language = yaml_data['language']
            if 'log_level' in yaml_data:
                config.log_level = yaml_data['log_level']
            if 'translation_log_max_bytes' in yaml_data:
                config.translation_log_max_bytes = int(yaml_data['translation_log_max_bytes'])
                
            logger.info(f"Configuration loaded from {resolved_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {resolved_path}: {e}")
    else:
        logger.warning(f"Config file not found at {resolved_path}, using defaults.")
    
    return config
