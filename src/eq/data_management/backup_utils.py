"""Backup helpers for contract migration and data normalization work."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class BackupArtifact:
    """Metadata for a created backup snapshot."""

    backup_root: Path
    manifest_files: Path
    manifest_sha256: Path
    manifest_meta: Path


def _sha256_for_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def create_backup_snapshot(
    sources: Sequence[Path],
    backup_dir: Path,
    label: str,
    timestamp: str | None = None,
) -> BackupArtifact:
    """Copy source trees into a timestamped backup directory with manifests."""
    stamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_root = backup_dir / f'{label}_{stamp}'
    snapshot_root.mkdir(parents=True, exist_ok=False)

    copied_roots: List[Path] = []
    for source in sources:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f'Backup source does not exist: {source}')
        destination = snapshot_root / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        copied_roots.append(destination)

    files: List[Path] = []
    for copied_root in copied_roots:
        if copied_root.is_dir():
            files.extend(sorted(path for path in copied_root.rglob('*') if path.is_file()))
        else:
            files.append(copied_root)

    manifest_files = snapshot_root / 'manifest.files'
    manifest_sha256 = snapshot_root / 'manifest.sha256'
    manifest_meta = snapshot_root / 'manifest.meta'

    with manifest_files.open('w', encoding='utf-8') as handle:
        for file_path in files:
            handle.write(f'{file_path.relative_to(snapshot_root)}\n')

    with manifest_sha256.open('w', encoding='utf-8') as handle:
        for file_path in files:
            checksum = _sha256_for_file(file_path)
            handle.write(f'{checksum}  {file_path.relative_to(snapshot_root)}\n')

    with manifest_meta.open('w', encoding='utf-8') as handle:
        handle.write(f'created_at={stamp}\n')
        handle.write(f'backup_root={snapshot_root}\n')
        handle.write(f'num_sources={len(sources)}\n')
        handle.write(f'num_files={len(files)}\n')

    return BackupArtifact(
        backup_root=snapshot_root,
        manifest_files=manifest_files,
        manifest_sha256=manifest_sha256,
        manifest_meta=manifest_meta,
    )
