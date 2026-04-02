"""Canonical filename parsing and validation for preeclampsia raw data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from eq.core.constants import IMAGE_EXTENSIONS, MASK_EXTENSIONS

CANONICAL_SUBJECT_IMAGE_RE = re.compile(r'^(?P<subject_prefix>T\d+)-(?P<image_index>\d+)$')
LEGACY_SUBJECT_IMAGE_RE = re.compile(r'^(?P<subject_prefix>T\d+)_Image(?P<legacy_index>\d+)$')


@dataclass(frozen=True)
class ParsedSubjectImageId:
    """Parsed representation of a canonical or legacy subject-image filename stem."""

    subject_prefix: str
    image_index: int
    subject_image_id: str
    naming_format: str
    original_stem: str


def _normalize_extension(suffix: str) -> str:
    return suffix.lower()


def is_supported_image_extension(path: Path) -> bool:
    """Return whether the path uses a supported raw image extension."""
    return _normalize_extension(path.suffix) in IMAGE_EXTENSIONS


def is_supported_mask_extension(path: Path) -> bool:
    """Return whether the path uses a supported raw mask extension."""
    return _normalize_extension(path.suffix) in MASK_EXTENSIONS


def parse_subject_image_stem(stem: str, allow_legacy: bool = True) -> Optional[ParsedSubjectImageId]:
    """Parse a raw image stem into a normalized subject-image identifier."""
    canonical_match = CANONICAL_SUBJECT_IMAGE_RE.match(stem)
    if canonical_match:
        subject_prefix = canonical_match.group('subject_prefix')
        image_index = int(canonical_match.group('image_index'))
        return ParsedSubjectImageId(
            subject_prefix=subject_prefix,
            image_index=image_index,
            subject_image_id=f'{subject_prefix}-{image_index}',
            naming_format='canonical',
            original_stem=stem,
        )

    if allow_legacy:
        legacy_match = LEGACY_SUBJECT_IMAGE_RE.match(stem)
        if legacy_match:
            subject_prefix = legacy_match.group('subject_prefix')
            legacy_index = int(legacy_match.group('legacy_index'))
            return ParsedSubjectImageId(
                subject_prefix=subject_prefix,
                image_index=legacy_index,
                subject_image_id=f'{subject_prefix}-{legacy_index}',
                naming_format='legacy',
                original_stem=stem,
            )

    return None


def parse_image_path(path: Path, allow_legacy: bool = True) -> Optional[ParsedSubjectImageId]:
    """Parse a raw image path."""
    if not is_supported_image_extension(path):
        return None
    return parse_subject_image_stem(path.stem, allow_legacy=allow_legacy)


def parse_mask_path(path: Path, allow_legacy: bool = True) -> Optional[ParsedSubjectImageId]:
    """Parse a raw mask path with a required ``_mask`` suffix."""
    if not is_supported_mask_extension(path):
        return None
    stem = path.stem
    if not stem.endswith('_mask'):
        return None
    return parse_subject_image_stem(stem[:-5], allow_legacy=allow_legacy)


def canonical_image_name(subject_image_id: str, suffix: str) -> str:
    """Return the canonical image filename for a subject-image id."""
    return f'{subject_image_id}{suffix.lower()}'


def canonical_mask_name(subject_image_id: str, suffix: str) -> str:
    """Return the canonical mask filename for a subject-image id."""
    return f'{subject_image_id}_mask{suffix.lower()}'


def subject_prefix_from_subject_image_id(subject_image_id: str) -> str:
    """Return the subject prefix (e.g. ``T19``) from a canonical id like ``T19-1``."""
    parsed = parse_subject_image_stem(subject_image_id, allow_legacy=False)
    if parsed is None:
        raise ValueError(f'Invalid canonical subject-image id: {subject_image_id}')
    return parsed.subject_prefix


def classify_naming_conventions(paths: Iterable[Path]) -> set[str]:
    """Return the naming conventions observed across the provided paths."""
    conventions: set[str] = set()
    for path in paths:
        parsed = parse_image_path(path, allow_legacy=True) or parse_mask_path(path, allow_legacy=True)
        if parsed is not None:
            conventions.add(parsed.naming_format)
    return conventions
