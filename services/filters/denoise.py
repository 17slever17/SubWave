from __future__ import annotations

import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

from services.runtime.paths import resolve_runtime_path

logger = logging.getLogger(__name__)
_USE_CONFIGURED_ATTEN_LIMIT = object()


def ensure_torchaudio_compat() -> None:
    try:
        import torchaudio
    except Exception:
        return

    if "torchaudio.backend.common" in sys.modules:
        return

    audio_metadata = getattr(torchaudio, "AudioMetaData", None)
    if audio_metadata is None:
        class AudioMetaData:  # pragma: no cover - compatibility shim
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        audio_metadata = AudioMetaData

    backend_pkg = sys.modules.get("torchaudio.backend")
    if backend_pkg is None:
        backend_pkg = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_pkg

    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = audio_metadata
    setattr(backend_pkg, "common", common_mod)
    sys.modules["torchaudio.backend.common"] = common_mod


def resample_mono_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    return soxr.resample(audio, src_sr, dst_sr, quality="HQ").astype(np.float32, copy=False)


def fit_audio_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size == target_len:
        return audio
    if audio.size > target_len:
        return audio[:target_len]
    if audio.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    return np.pad(audio, (0, target_len - audio.size)).astype(np.float32)


class BaseDenoiser:
    enabled = False

    def process(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.float32)

    def process_window(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        atten_lim_db: float | None | object = _USE_CONFIGURED_ATTEN_LIMIT,
    ) -> np.ndarray:
        return np.asarray(samples, dtype=np.float32)


class PassthroughDenoiser(BaseDenoiser):
    enabled = False


@dataclass
class FastEnhancerSettings:
    model_path: str = "models/fastenhancer/fastenhancer_b_48khz.onnx"
    model_sample_rate: int = 48000
    n_fft: int = 1024
    hop_samples: int = 512
    num_threads: int = 4
    wet_mix: float = 0.5


class FastEnhancerDenoiser(BaseDenoiser):
    enabled = True

    def __init__(self, settings: FastEnhancerSettings):
        import onnxruntime as ort

        self.settings = settings
        model_path = Path(settings.model_path)
        if not model_path.is_absolute():
            model_path = resolve_runtime_path(model_path) or model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"FastEnhancer model not found: {model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(settings.num_threads))
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_specs = self._session.get_inputs()
        self._output_names = [
            output.name for output in self._session.get_outputs()
        ]
        logger.info(
            "FastEnhancer denoiser initialized: model=%s sr=%s wet_mix=%.2f",
            model_path,
            settings.model_sample_rate,
            settings.wet_mix,
        )

    def process_window(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        atten_lim_db: float | None | object = _USE_CONFIGURED_ATTEN_LIMIT,
    ) -> np.ndarray:
        del atten_lim_db
        original = np.asarray(samples, dtype=np.float32).reshape(-1)
        if original.size == 0:
            return original

        source_sr = int(sample_rate or self.settings.model_sample_rate)
        model_audio = resample_mono_audio(
            original,
            source_sr,
            self.settings.model_sample_rate,
        )
        padded_length = (
            (model_audio.size + self.settings.hop_samples - 1)
            // self.settings.hop_samples
            * self.settings.hop_samples
        )
        padded = fit_audio_length(model_audio, padded_length)
        caches = {
            spec.name: np.zeros(spec.shape, dtype=np.float32)
            for spec in self._input_specs[1:]
        }
        delay_samples = self.settings.n_fft - self.settings.hop_samples
        flush_hops = (
            delay_samples + self.settings.hop_samples - 1
        ) // self.settings.hop_samples
        stream = np.concatenate(
            (
                padded,
                np.zeros(
                    flush_hops * self.settings.hop_samples,
                    dtype=np.float32,
                ),
            )
        )
        outputs: list[np.ndarray] = []
        for start in range(0, stream.size, self.settings.hop_samples):
            feeds = {
                self._input_specs[0].name: stream[
                    start : start + self.settings.hop_samples
                ][None, :],
                **caches,
            }
            result = self._session.run(self._output_names, feeds)
            outputs.append(
                np.asarray(result[0], dtype=np.float32).reshape(-1)
            )
            caches = {
                spec.name: np.asarray(value, dtype=np.float32)
                for spec, value in zip(self._input_specs[1:], result[1:])
            }
        enhanced = np.concatenate(outputs)[
            delay_samples : delay_samples + model_audio.size
        ]
        enhanced = resample_mono_audio(
            enhanced,
            self.settings.model_sample_rate,
            source_sr,
        )
        enhanced = fit_audio_length(enhanced, original.size)
        wet = float(np.clip(self.settings.wet_mix, 0.0, 1.0))
        mixed = (enhanced * wet) + (original * (1.0 - wet))
        return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)


@dataclass
class DeepFilterNet2Settings:
    model_name: str = "DeepFilterNet2"
    model_sample_rate: int = 48000
    block_ms: int = 40
    device: str = "cpu"
    post_filter: bool = False
    atten_lim_db: float | None = None
    wet_mix: float = 1.0
    log_level: str = "ERROR"


class DeepFilterNet2Denoiser(BaseDenoiser):
    enabled = True

    def __init__(self, input_sr: int, settings: DeepFilterNet2Settings):
        self.input_sr = int(input_sr)
        self.settings = settings
        self._input_block_samples = max(1, int(round(self.input_sr * self.settings.block_ms / 1000.0)))
        self._input_carry = np.zeros(0, dtype=np.float32)

        os.environ["DEVICE"] = str(settings.device or "cpu").lower()
        ensure_torchaudio_compat()

        import torch
        from df.enhance import df_features, enhance, init_df
        from df.model import ModelParams
        from df.utils import as_complex, get_device

        self._torch = torch
        self._df_features = df_features
        self._enhance = enhance
        self._as_complex = as_complex
        self._get_device = get_device

        self._model, self._df_state, _ = init_df(
            model_base_dir=settings.model_name,
            post_filter=settings.post_filter,
            log_level=settings.log_level,
            log_file=None,
            config_allow_defaults=True,
            default_model=settings.model_name,
        )
        self._model.eval()
        self._device = get_device()
        self.df_sr = int(settings.model_sample_rate or ModelParams().sr)
        self.nb_df = getattr(self._model, "nb_df", getattr(self._model, "df_bins", ModelParams().nb_df))
        if hasattr(self._model, "reset_h0"):
            self._model.reset_h0(batch_size=1, device=self._device)

        logger.info(
            "DeepFilterNet2 denoiser initialized: model=%s device=%s df_sr=%s input_sr=%s block_ms=%s wet_mix=%.2f",
            settings.model_name,
            self._device,
            self.df_sr,
            self.input_sr,
            self.settings.block_ms,
            settings.wet_mix,
        )

    def _process_block(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio

        original = audio
        if self.input_sr != self.df_sr:
            audio = resample_mono_audio(audio, self.input_sr, self.df_sr)

        audio_t = self._torch.from_numpy(audio).unsqueeze(0)
        spec, erb_feat, spec_feat = self._df_features(
            audio_t,
            self._df_state,
            self.nb_df,
            device=self._device,
        )

        with self._torch.no_grad():
            enhanced = self._model(spec.clone(), erb_feat, spec_feat)[0].cpu()

        enhanced = self._as_complex(enhanced.squeeze(1))
        if self.settings.atten_lim_db is not None and abs(self.settings.atten_lim_db) > 0:
            lim = 10 ** (-abs(self.settings.atten_lim_db) / 20)
            enhanced = self._as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)

        enhanced_audio = self._torch.as_tensor(self._df_state.synthesis(enhanced.numpy())).squeeze(0).numpy()
        if self.df_sr != self.input_sr:
            enhanced_audio = resample_mono_audio(enhanced_audio, self.df_sr, self.input_sr)

        enhanced_audio = fit_audio_length(enhanced_audio, original.size)
        wet = float(np.clip(self.settings.wet_mix, 0.0, 1.0))
        if wet < 1.0:
            enhanced_audio = (enhanced_audio * wet) + (original * (1.0 - wet))

        return np.clip(enhanced_audio.astype(np.float32, copy=False), -1.0, 1.0)

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return samples

        pending = np.concatenate([self._input_carry, samples], axis=0) if self._input_carry.size else samples
        processed_blocks: list[np.ndarray] = []
        while pending.size >= self._input_block_samples:
            block = pending[: self._input_block_samples]
            pending = pending[self._input_block_samples :]
            processed = self._process_block(block)
            processed_blocks.append(processed)
        self._input_carry = pending.astype(np.float32, copy=False)
        if not processed_blocks:
            return np.zeros(0, dtype=np.float32)
        if len(processed_blocks) == 1:
            return processed_blocks[0]
        return np.concatenate(processed_blocks, axis=0)

    def process_window(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        atten_lim_db: float | None | object = _USE_CONFIGURED_ATTEN_LIMIT,
    ) -> np.ndarray:
        original = np.asarray(samples, dtype=np.float32).reshape(-1)
        if original.size == 0:
            return original

        source_sr = int(sample_rate or self.input_sr)
        audio = resample_mono_audio(original, source_sr, self.df_sr)
        audio_t = self._torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        # The official whole-buffer path supplies temporal context and delay
        # compensation. Reset both recurrent and STFT state between windows.
        self._df_state.reset()
        effective_atten_lim_db = (
            self.settings.atten_lim_db
            if atten_lim_db is _USE_CONFIGURED_ATTEN_LIMIT
            else atten_lim_db
        )
        with self._torch.no_grad():
            enhanced_t = self._enhance(
                self._model,
                self._df_state,
                audio_t,
                pad=True,
                atten_lim_db=effective_atten_lim_db,
            )
        enhanced = enhanced_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        enhanced = resample_mono_audio(enhanced, self.df_sr, source_sr)
        enhanced = fit_audio_length(enhanced, original.size)

        wet = float(np.clip(self.settings.wet_mix, 0.0, 1.0))
        if wet < 1.0:
            enhanced = (enhanced * wet) + (original * (1.0 - wet))
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)


def build_denoiser(config, sample_rate: int) -> BaseDenoiser:
    audio_conf = getattr(config, "audio", config)
    dfn2_conf = getattr(audio_conf, "dfn2", None)
    enabled = bool(getattr(dfn2_conf, "enabled", False))
    if not enabled:
        return PassthroughDenoiser()

    vad_atten_lim_db = getattr(
        dfn2_conf,
        "vad_atten_lim_db",
        getattr(dfn2_conf, "atten_lim_db", None),
    )
    settings = DeepFilterNet2Settings(
        model_name=str(getattr(dfn2_conf, "model", "DeepFilterNet2")),
        model_sample_rate=int(getattr(dfn2_conf, "input_sample_rate", 48000)),
        block_ms=int(getattr(dfn2_conf, "block_ms", 40)),
        device=str(getattr(dfn2_conf, "device", "cpu")),
        post_filter=bool(getattr(dfn2_conf, "post_filter", False)),
        atten_lim_db=vad_atten_lim_db,
        wet_mix=float(getattr(dfn2_conf, "wet_mix", 1.0)),
        log_level=str(getattr(dfn2_conf, "log_level", "ERROR")),
    )
    try:
        return DeepFilterNet2Denoiser(sample_rate, settings)
    except Exception as exc:
        logger.exception("DeepFilterNet2 denoiser initialization failed, falling back to passthrough: %s", exc)
        return PassthroughDenoiser()


def build_stt_denoiser(
    config,
    sample_rate: int,
    language: str,
    dfn_fallback: BaseDenoiser | None = None,
) -> BaseDenoiser:
    audio_conf = getattr(config, "audio", config)
    fast_conf = getattr(audio_conf, "fastenhancer", None)
    language_key = str(language).strip().lower().split("-", 1)[0]
    supported = {
        str(item).strip().lower()
        for item in getattr(
            fast_conf,
            "languages",
            ("en", "ja", "ko", "es"),
        )
    }
    if not bool(getattr(fast_conf, "enabled", True)):
        return dfn_fallback or build_denoiser(config, sample_rate)
    if language_key not in supported:
        return dfn_fallback or build_denoiser(config, sample_rate)

    wet_mix_by_language = getattr(fast_conf, "wet_mix_by_language", {})
    wet_mix = float(
        wet_mix_by_language.get(
            language_key,
            getattr(fast_conf, "wet_mix", 0.5),
        )
    )
    settings = FastEnhancerSettings(
        model_path=str(
            getattr(
                fast_conf,
                "model_path",
                "models/fastenhancer/fastenhancer_b_48khz.onnx",
            )
        ),
        model_sample_rate=int(
            getattr(fast_conf, "input_sample_rate", 48000)
        ),
        n_fft=int(getattr(fast_conf, "n_fft", 1024)),
        hop_samples=int(getattr(fast_conf, "hop_samples", 512)),
        num_threads=int(getattr(fast_conf, "num_threads", 4)),
        wet_mix=wet_mix,
    )
    try:
        return FastEnhancerDenoiser(settings)
    except Exception as exc:
        logger.exception(
            "FastEnhancer initialization failed for language=%s; "
            "falling back to DeepFilterNet: %s",
            language_key,
            exc,
        )
        return dfn_fallback or build_denoiser(config, sample_rate)
