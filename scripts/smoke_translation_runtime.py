from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = APP_DIR.parent
for path in (APP_DIR, PARENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.settings import Config  # noqa: E402
from services.llm.translator import LLMTranslator  # noqa: E402


MODELS = {
    "e2b": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
    "e4b": "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real GGUF translation through the configured backend."
    )
    parser.add_argument("--model", choices=MODELS, default="e2b")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument(
        "--text",
        default="I thought she had already sent it, but apparently she forgot.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = Config()
    config.language = "en"
    config.translation.model_path = MODELS[args.model]
    config.translation.device = args.device
    config.translation.n_gpu_layers = -1 if args.device == "gpu" else 0
    config.translation.mtp_enabled = not args.no_mtp
    config.translation.context_subtitles = 0
    config.translation.n_ctx = 544

    started = time.perf_counter()
    translator = LLMTranslator(config)
    load_seconds = time.perf_counter() - started
    if translator.llm is None:
        print("[smoke] model failed to load")
        return 1

    try:
        generated = time.perf_counter()
        output = translator.translate(args.text)
        generation_seconds = time.perf_counter() - generated
        backend = type(translator.llm).__name__
        print(f"[smoke] backend={backend}")
        print(f"[smoke] load_seconds={load_seconds:.3f}")
        print(f"[smoke] generation_seconds={generation_seconds:.3f}")
        print(f"[smoke] source={args.text}")
        print(f"[smoke] translation={output}")
        if not output.strip() or output.strip() == args.text.strip():
            print("[smoke] translation was empty or unchanged")
            return 2
        return 0
    finally:
        translator.close()


if __name__ == "__main__":
    raise SystemExit(main())
