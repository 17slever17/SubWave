from __future__ import annotations

from difflib import SequenceMatcher
from collections import Counter
import re
import unicodedata


_EDGE_PUNCTUATION = " \t\r\n.,!?;:。！？、…-—–|/\\()[]{}\"'«»"


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def text_preview(text: str, max_chars: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}... <{len(compact)} chars>"


def _similarity(left: str, right: str) -> float:
    left_normalized = normalize_for_comparison(left)
    right_normalized = normalize_for_comparison(right)
    if not left_normalized or not right_normalized:
        return 0.0
    length_ratio = min(len(left_normalized), len(right_normalized)) / max(
        len(left_normalized), len(right_normalized)
    )
    if length_ratio < 0.72:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def text_similarity(left: str, right: str) -> float:
    return _similarity(left, right)


def is_context_leak(
    current_source: str,
    previous_source: str,
    translation: str,
    previous_translation: str,
    translation_similarity_threshold: float = 0.68,
    source_similarity_limit: float = 0.55,
) -> bool:
    if not all((current_source, previous_source, translation, previous_translation)):
        return False
    if text_similarity(current_source, previous_source) >= source_similarity_limit:
        return False
    return text_similarity(translation, previous_translation) >= translation_similarity_threshold


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = re.findall(r"[^\W_]+", left.casefold(), flags=re.UNICODE)
    right_tokens = re.findall(r"[^\W_]+", right.casefold(), flags=re.UNICODE)
    if not left_tokens or not right_tokens:
        return 0.0
    common = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    return (2.0 * common) / (len(left_tokens) + len(right_tokens))


def source_repetition_similarity(
    previous_source: str,
    current_source: str,
    source_language: str = "",
) -> float:
    previous = previous_source.strip()
    current = current_source.strip()
    if not previous or not current:
        return 0.0

    previous_normalized = normalize_for_comparison(previous)
    current_normalized = normalize_for_comparison(current)
    if not previous_normalized or not current_normalized:
        return 0.0
    if current_normalized.startswith(previous_normalized):
        return 1.0

    no_space_language = source_language.strip().lower() in {"ja", "zh", "ko", "th"}
    first_sentence = re.split(r"(?<=[.!?。！？…])", current, maxsplit=1)[0]
    candidates = [current, first_sentence]

    if no_space_language:
        prefix = current_normalized[: len(previous_normalized)]
        direct_prefix_score = SequenceMatcher(None, previous_normalized, prefix).ratio()
        return max(direct_prefix_score, *(text_similarity(previous, candidate) for candidate in candidates))

    previous_tokens = re.findall(r"[^\W_]+", previous, flags=re.UNICODE)
    current_tokens = re.findall(r"[^\W_]+", current, flags=re.UNICODE)
    token_count = len(previous_tokens)
    for extra_tokens in (-1, 0, 1, 2):
        prefix_count = token_count + extra_tokens
        if prefix_count > 0:
            candidates.append(" ".join(current_tokens[:prefix_count]))

    return max(
        max(text_similarity(previous, candidate), _token_overlap_similarity(previous, candidate))
        for candidate in candidates
        if candidate
    )


def strip_repeated_subtitle_prefix(
    previous_translation: str,
    current_translation: str,
    previous_source: str = "",
    current_source: str = "",
    source_language: str = "",
    similarity_threshold: float = 0.82,
    min_duplicate_chars: int = 12,
    source_repeat_similarity_threshold: float = 0.78,
    cjk_source_repeat_similarity_threshold: float = 0.90,
) -> tuple[str, bool, float]:
    previous = previous_translation.strip()
    current = current_translation.strip()
    previous_length = len(normalize_for_comparison(previous))
    if previous_length < min_duplicate_chars or not current:
        return current, False, 0.0

    min_split = max(1, int(len(previous) * 0.55))
    max_split = min(len(current), int(len(previous) * 1.55) + 20)
    boundaries = {len(current)}
    for index in range(min_split, max_split + 1):
        if index == len(current) or current[index - 1].isspace() or current[index - 1] in ".!?。！？…":
            boundaries.add(index)

    best: tuple[int, float, int, int] | None = None
    for split_at in boundaries:
        if split_at < min_split or split_at > max_split:
            continue
        prefix = current[:split_at].strip()
        prefix_length = len(normalize_for_comparison(prefix))
        if prefix_length < min_duplicate_chars:
            continue
        character_score = text_similarity(previous, prefix)
        token_score = _token_overlap_similarity(previous, prefix)
        similarity = max(character_score, token_score * 0.97)
        if similarity < similarity_threshold:
            continue
        length_penalty = abs(prefix_length - previous_length) / max(previous_length, 1) * 0.04
        is_terminal_boundary = prefix.rstrip().endswith((".", "!", "?", "…", "。", "！", "？"))
        terminal_bonus = 0.01 if is_terminal_boundary else 0.0
        rank = similarity - length_penalty + terminal_bonus
        candidate = (int(is_terminal_boundary), rank, split_at, int(similarity * 10_000))
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return current, False, 0.0

    source_similarity = source_repetition_similarity(
        previous_source=previous_source,
        current_source=current_source,
        source_language=source_language,
    )
    source_threshold = (
        cjk_source_repeat_similarity_threshold
        if source_language.strip().lower() in {"ja", "zh", "ko", "th"}
        else source_repeat_similarity_threshold
    )
    if source_similarity >= source_threshold:
        return current, False, best[3] / 10_000.0

    _, _, split_at, similarity_scaled = best
    remainder = current[split_at:].lstrip(" \t\r\n.,!?;:。！？、…-—–")
    return remainder, True, similarity_scaled / 10_000.0


def collapse_repeated_text(
    text: str,
    similarity_threshold: float = 0.90,
    min_repeat_chars: int = 8,
) -> tuple[str, bool]:
    """Collapse an accidental duplicated response while preserving short emphasis."""
    result = re.sub(r"\s+", " ", text).strip()
    if len(normalize_for_comparison(result)) < min_repeat_chars * 2:
        return result, False

    changed = False
    for _ in range(3):
        candidate = _collapse_once(result, similarity_threshold, min_repeat_chars)
        if candidate == result:
            break
        result = candidate
        changed = True
    return result, changed


def cap_repeated_token_sequences(
    text: str,
    max_repetitions: int = 3,
    max_block_tokens: int = 16,
) -> tuple[str, bool]:
    """Keep natural repetition while capping runaway ASR token loops."""
    result = re.sub(r"\s+", " ", text).strip()
    if not result or max_repetitions < 1:
        return result, False

    if " " not in result and re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", result):
        return _cap_repeated_character_sequences(result, max_repetitions)

    raw_tokens = result.split(" ")
    normalized_tokens = [
        "".join(character.casefold() for character in token if character.isalnum())
        for token in raw_tokens
    ]
    changed = False

    max_block = min(max_block_tokens, len(raw_tokens) // (max_repetitions + 1))
    for block_size in range(max_block, 0, -1):
        start = 0
        while start + block_size * (max_repetitions + 1) <= len(raw_tokens):
            block = normalized_tokens[start : start + block_size]
            if not all(block):
                start += 1
                continue

            repeat_count = 1
            while (
                start + (repeat_count + 1) * block_size <= len(raw_tokens)
                and normalized_tokens[
                    start + repeat_count * block_size :
                    start + (repeat_count + 1) * block_size
                ]
                == block
            ):
                repeat_count += 1

            if repeat_count <= max_repetitions:
                start += 1
                continue

            remove_start = start + max_repetitions * block_size
            remove_end = start + repeat_count * block_size
            del raw_tokens[remove_start:remove_end]
            del normalized_tokens[remove_start:remove_end]
            changed = True

    return " ".join(raw_tokens), changed


def _cap_repeated_character_sequences(
    text: str,
    max_repetitions: int,
    max_unit_chars: int = 24,
) -> tuple[str, bool]:
    characters = list(text)
    changed = False
    max_unit = min(max_unit_chars, len(characters) // (max_repetitions + 1))

    for unit_length in range(max_unit, 0, -1):
        start = 0
        while start + unit_length * (max_repetitions + 1) <= len(characters):
            unit = characters[start : start + unit_length]
            repeat_count = 1
            while (
                start + (repeat_count + 1) * unit_length <= len(characters)
                and characters[
                    start + repeat_count * unit_length :
                    start + (repeat_count + 1) * unit_length
                ]
                == unit
            ):
                repeat_count += 1

            if repeat_count <= max_repetitions:
                start += 1
                continue

            remove_start = start + max_repetitions * unit_length
            remove_end = start + repeat_count * unit_length
            del characters[remove_start:remove_end]
            changed = True

    return "".join(characters), changed


def _collapse_once(text: str, similarity_threshold: float, min_repeat_chars: int) -> str:
    normalized_length = len(normalize_for_comparison(text))
    for repeat_count in range(4, 1, -1):
        if len(text) % repeat_count:
            continue
        chunk_length = len(text) // repeat_count
        chunks = [text[index * chunk_length : (index + 1) * chunk_length] for index in range(repeat_count)]
        if len(normalize_for_comparison(chunks[0])) < min_repeat_chars:
            continue
        if all(_similarity(chunks[0], chunk) >= similarity_threshold for chunk in chunks[1:]):
            return chunks[0].strip(_EDGE_PUNCTUATION).strip()

    midpoint = len(text) // 2
    search_radius = max(2, int(len(text) * 0.12))
    candidates: list[tuple[float, int, str]] = []
    for split_at in range(max(1, midpoint - search_radius), min(len(text), midpoint + search_radius + 1)):
        left = text[:split_at].strip(_EDGE_PUNCTUATION)
        right = text[split_at:].strip(_EDGE_PUNCTUATION)
        if min(len(normalize_for_comparison(left)), len(normalize_for_comparison(right))) < min_repeat_chars:
            continue
        similarity = _similarity(left, right)
        if similarity < similarity_threshold:
            continue
        is_boundary = text[split_at - 1] in _EDGE_PUNCTUATION or text[split_at] in _EDGE_PUNCTUATION
        is_exact_without_boundary = similarity == 1.0
        if is_boundary or is_exact_without_boundary:
            candidates.append((similarity, -abs(split_at - midpoint), left.strip()))

    if candidates:
        return max(candidates)[2]

    return text


def has_repeated_character_loop(
    text: str,
    min_repetitions: int = 4,
    max_unit_chars: int = 24,
    min_repeated_chars: int = 12,
) -> bool:
    normalized = normalize_for_comparison(text)
    if len(normalized) < min_repeated_chars:
        return False
    max_start = min(64, len(normalized) - min_repeated_chars)
    for start in range(max_start + 1):
        max_unit = min(max_unit_chars, (len(normalized) - start) // min_repetitions)
        for unit_length in range(1, max_unit + 1):
            unit = normalized[start : start + unit_length]
            repeat_count = 1
            cursor = start + unit_length
            while normalized[cursor : cursor + unit_length] == unit:
                repeat_count += 1
                cursor += unit_length
            repeated_chars = repeat_count * unit_length
            if repeat_count >= min_repetitions and repeated_chars >= min_repeated_chars:
                return True
    return False


def has_repeated_token_loop(text: str, min_block_tokens: int = 1, repetitions: int = 3) -> bool:
    if has_repeated_character_loop(text):
        return True
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < min_block_tokens * repetitions:
        return False
    max_block_tokens = min(16, len(tokens) // repetitions)
    for block_size in range(min_block_tokens, max_block_tokens + 1):
        required_repetitions = 5 if block_size == 1 else repetitions
        for start in range(0, len(tokens) - block_size * required_repetitions + 1):
            block = tokens[start : start + block_size]
            repeat_count = 1
            while (
                start + (repeat_count + 1) * block_size <= len(tokens)
                and tokens[
                    start + repeat_count * block_size :
                    start + (repeat_count + 1) * block_size
                ] == block
            ):
                repeat_count += 1
            repeated_coverage = repeat_count * block_size / len(tokens)
            if repeat_count >= required_repetitions and (
                repeated_coverage >= 0.6 or repeat_count >= 8
            ):
                return True
    return False
