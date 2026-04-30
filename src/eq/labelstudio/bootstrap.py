"""Local Label Studio bootstrap for glomerulus-instance grading."""

from __future__ import annotations

import http.client
import json
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from eq.utils.paths import get_active_runtime_root, get_repo_root

SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
DEFAULT_PROJECT_TITLE = 'EQ Glomerulus Grading'
DEFAULT_CONTAINER_NAME = 'eq-labelstudio'
DEFAULT_DOCKER_IMAGE = 'heartexlabs/label-studio:latest'
DEFAULT_USERNAME = 'eq-admin@example.local'
DEFAULT_PASSWORD = 'eq-labelstudio'
DEFAULT_API_TOKEN = 'eq-local-token'
DEFAULT_PORT = 8080
DEFAULT_TIMEOUT_SECONDS = 60
DOCKER_DESKTOP_APP_PATH = Path('/Applications/Docker.app')
CONTAINER_LOCAL_FILES_ROOT = '/label-studio/media'


class BootstrapError(RuntimeError):
    """Raised when local Label Studio bootstrap cannot proceed."""


@dataclass(frozen=True)
class BootstrapPlan:
    images_dir: Path
    runtime_root: Path
    data_dir: Path
    media_dir: Path
    imports_dir: Path
    bootstrap_dir: Path
    task_manifest_path: Path
    label_config_path: Path
    project_title: str
    url: str
    project_url: str
    container_name: str
    docker_image: str
    username: str
    password: str
    api_token: str
    timeout_seconds: int
    image_paths: tuple[Path, ...]
    task_count: int


@dataclass(frozen=True)
class BootstrapResult:
    plan: BootstrapPlan
    task_manifest_path: Path
    project_url: str
    message: str


DockerRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def discover_image_files(images_dir: Path) -> list[Path]:
    """Return supported image files under `images_dir` in deterministic order."""
    images_dir = Path(images_dir).expanduser()
    if not images_dir.exists() or not images_dir.is_dir():
        raise BootstrapError(f'Image directory does not exist: {images_dir}')
    return [
        path
        for path in sorted(images_dir.rglob('*'))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]


def build_labelstudio_tasks(
    images_dir: Path, image_paths: Iterable[Path]
) -> list[dict[str, Any]]:
    """Build Label Studio local-file tasks for image paths."""
    images_dir = Path(images_dir).expanduser().resolve()
    tasks: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_path = Path(image_path).expanduser().resolve()
        relative = image_path.relative_to(images_dir).as_posix()
        parts = Path(relative).parts
        subject_hint = parts[0] if len(parts) > 1 else image_path.stem
        tasks.append(
            {
                'data': {
                    'image': f'/data/local-files/?d={relative}',
                    'source_relative_path': relative,
                    'source_filename': image_path.name,
                    'subject_hint': subject_hint,
                }
            }
        )
    return tasks


def plan_bootstrap(
    *,
    images_dir: Path,
    runtime_root: Path | None = None,
    project_name: str = DEFAULT_PROJECT_TITLE,
    port: int = DEFAULT_PORT,
    container_name: str = DEFAULT_CONTAINER_NAME,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    api_token: str = DEFAULT_API_TOKEN,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> BootstrapPlan:
    """Create a complete bootstrap plan without side effects."""
    images_dir = Path(images_dir).expanduser().resolve()
    image_paths = tuple(discover_image_files(images_dir))
    root = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else get_active_runtime_root() / 'labelstudio'
    )
    url = f'http://localhost:{int(port)}'
    safe_project = _safe_slug(project_name)
    imports_dir = root / 'imports'
    return BootstrapPlan(
        images_dir=images_dir,
        runtime_root=root,
        data_dir=root / 'data',
        media_dir=images_dir,
        imports_dir=imports_dir,
        bootstrap_dir=root / 'bootstrap',
        task_manifest_path=imports_dir / f'{safe_project}_tasks.json',
        label_config_path=get_repo_root() / 'configs' / 'label_studio_glomerulus_grading.xml',
        project_title=project_name,
        url=url,
        project_url='',
        container_name=container_name,
        docker_image=docker_image,
        username=username,
        password=password,
        api_token=api_token,
        timeout_seconds=int(timeout_seconds),
        image_paths=image_paths,
        task_count=len(image_paths),
    )


def run_bootstrap(
    *,
    images_dir: Path,
    runtime_root: Path | None = None,
    project_name: str = DEFAULT_PROJECT_TITLE,
    port: int = DEFAULT_PORT,
    container_name: str = DEFAULT_CONTAINER_NAME,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    api_token: str = DEFAULT_API_TOKEN,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    docker_runner: DockerRunner | None = None,
    api_client: 'LabelStudioApiClient | Any | None' = None,
) -> BootstrapResult:
    """Prepare a local Label Studio project and import image tasks."""
    plan = plan_bootstrap(
        images_dir=images_dir,
        runtime_root=runtime_root,
        project_name=project_name,
        port=port,
        container_name=container_name,
        docker_image=docker_image,
        username=username,
        password=password,
        api_token=api_token,
        timeout_seconds=timeout_seconds,
    )
    tasks = build_labelstudio_tasks(plan.images_dir, plan.image_paths)
    _write_task_manifest(plan, tasks)
    if dry_run:
        return BootstrapResult(
            plan=plan,
            task_manifest_path=plan.task_manifest_path,
            project_url='',
            message=(
                f'Dry run: would start {plan.container_name}, create project '
                f'{plan.project_title!r}, and import {plan.task_count} images.'
            ),
        )

    runner = docker_runner or _run_command
    ensure_docker_available(runner, timeout_seconds=plan.timeout_seconds)
    start_labelstudio_container(plan, runner)
    client = api_client or LabelStudioApiClient(plan.url, plan.api_token)
    project_id = client.bootstrap_project(
        project_title=plan.project_title,
        label_config=plan.label_config_path.read_text(encoding='utf-8'),
        tasks=tasks,
        timeout_seconds=plan.timeout_seconds,
    )
    project_url = f'{plan.url}/projects/{project_id}/data'
    return BootstrapResult(
        plan=plan,
        task_manifest_path=plan.task_manifest_path,
        project_url=project_url,
        message=(
            f'Label Studio project ready: {project_url} '
            f'({plan.task_count} image tasks imported)'
        ),
    )


def ensure_docker_available(
    runner: DockerRunner | None = None, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
) -> None:
    """Fail early if Docker is missing, and start Docker Desktop on macOS when possible."""
    if shutil.which('docker') is None:
        raise BootstrapError(_docker_install_message())
    runner = runner or _run_command
    if runner(['docker', 'info']).returncode == 0:
        return
    if _can_start_docker_desktop():
        start = runner(['open', '-a', 'Docker'])
        if start.returncode != 0:
            raise BootstrapError(
                'Docker Desktop is installed but could not be started. Open Docker Desktop manually and rerun the command.'
            )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if runner(['docker', 'info']).returncode == 0:
                return
            time.sleep(2)
        raise BootstrapError(
            'Docker Desktop was started but the Docker daemon did not become ready. '
            'Open Docker Desktop and wait for it to finish starting, then rerun the command.'
        )
    raise BootstrapError(
        'Docker is installed but the daemon is not running. Start Docker and rerun the command.'
    )


def start_labelstudio_container(plan: BootstrapPlan, runner: DockerRunner | None = None) -> None:
    """Start or reuse the local Label Studio Docker container."""
    runner = runner or _run_command
    inspect = runner(['docker', 'inspect', plan.container_name])
    if inspect.returncode == 0:
        if not _container_matches_plan(inspect.stdout):
            remove = runner(['docker', 'rm', '-f', plan.container_name])
            if remove.returncode != 0:
                raise BootstrapError(f'Failed to replace container {plan.container_name}')
            command = docker_run_command(plan)
            result = runner(command)
            if result.returncode != 0:
                raise BootstrapError(f'Failed to start Label Studio Docker container: {result.stderr}')
            return
        start = runner(['docker', 'start', plan.container_name])
        if start.returncode != 0:
            raise BootstrapError(f'Failed to start container {plan.container_name}')
        return

    plan.data_dir.mkdir(parents=True, exist_ok=True)
    command = docker_run_command(plan)
    result = runner(command)
    if result.returncode != 0:
        raise BootstrapError(f'Failed to start Label Studio Docker container: {result.stderr}')


def docker_run_command(plan: BootstrapPlan) -> list[str]:
    """Return the Docker command used to start Label Studio."""
    return [
        'docker',
        'run',
        '-d',
        '--name',
        plan.container_name,
        '-p',
        f'{_port_from_url(plan.url)}:8080',
        '-v',
        f'{plan.data_dir}:/label-studio/data',
        '-v',
        f'{plan.media_dir}:{CONTAINER_LOCAL_FILES_ROOT}:ro',
        '-e',
        'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true',
        '-e',
        f'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={CONTAINER_LOCAL_FILES_ROOT}',
        '-e',
        f'LABEL_STUDIO_USERNAME={plan.username}',
        '-e',
        f'LABEL_STUDIO_PASSWORD={plan.password}',
        '-e',
        f'LABEL_STUDIO_USER_TOKEN={plan.api_token}',
        '-e',
        'LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true',
        plan.docker_image,
    ]


def _container_matches_plan(inspect_stdout: str) -> bool:
    try:
        details = json.loads(inspect_stdout)
    except json.JSONDecodeError:
        return False
    container = details[0] if isinstance(details, list) and details else {}
    mounts = container.get('Mounts', []) if isinstance(container, dict) else []
    destinations = {mount.get('Destination') for mount in mounts if isinstance(mount, dict)}
    return (
        CONTAINER_LOCAL_FILES_ROOT in destinations
        and f'{CONTAINER_LOCAL_FILES_ROOT}/images' not in destinations
    )


def _can_start_docker_desktop() -> bool:
    return platform.system() == 'Darwin' and DOCKER_DESKTOP_APP_PATH.exists()


def _docker_install_message() -> str:
    if platform.system() == 'Darwin':
        return (
            'Docker Desktop is not installed. Install it with '
            '`brew install --cask docker`, open Docker Desktop once, then rerun this command.'
        )
    return 'Docker executable not found. Install Docker or use --dry-run.'


class LabelStudioApiClient:
    """Small stdlib HTTP client for the Label Studio bootstrap API calls."""

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token

    def bootstrap_project(
        self,
        *,
        project_title: str,
        label_config: str,
        tasks: list[dict[str, Any]],
        timeout_seconds: int,
    ) -> int:
        self.wait_until_ready(timeout_seconds)
        project_id = self._find_project(project_title)
        if project_id is None:
            response = self._request_json(
                'POST',
                '/api/projects/',
                {'title': project_title, 'label_config': label_config},
            )
            project_id = int(response['id'])
        else:
            self._request_json(
                'PATCH',
                f'/api/projects/{project_id}/',
                {'label_config': label_config},
            )
        self._ensure_local_file_storages(project_id, tasks)
        self._request_json('POST', f'/api/projects/{project_id}/import', tasks)
        return project_id

    def wait_until_ready(self, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                self._request_json('GET', '/api/projects/')
                return
            except (BootstrapError, http.client.RemoteDisconnected, urllib.error.URLError):
                time.sleep(1)
        raise BootstrapError(
            f'Label Studio did not become reachable at {self.base_url} within '
            f'{timeout_seconds}s. Inspect the Docker container logs.'
        )

    def _find_project(self, project_title: str) -> int | None:
        response = self._request_json('GET', '/api/projects/')
        projects = _list_response_items(response)
        for project in projects:
            if project.get('title') == project_title:
                return int(project['id'])
        return None

    def _ensure_local_file_storages(self, project_id: int, tasks: list[dict[str, Any]]) -> None:
        response = self._request_json('GET', f'/api/storages/localfiles/?project={project_id}')
        storages = _list_response_items(response)
        existing_paths = {storage.get('path') for storage in storages}
        for storage_path in _local_file_storage_paths(tasks):
            if storage_path in existing_paths:
                continue
            self._request_json(
                'POST',
                '/api/storages/localfiles/',
                {
                    'project': project_id,
                    'path': storage_path,
                    'title': f'EQ Image Media {Path(storage_path).name}',
                    'use_blob_urls': True,
                    'recursive_scan': True,
                },
            )

    def _request_json(
        self, method: str, path: str, payload: Any | None = None
    ) -> Any:
        data = None
        headers = {'Authorization': f'Token {self.api_token}'}
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(
            f'{self.base_url}{path}', data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            raise BootstrapError(
                f'Label Studio API {method} {path} failed: {exc.code} {body}'
            ) from exc
        return json.loads(raw) if raw else {}


def _list_response_items(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        results = response.get('results')
        return results if isinstance(results, list) else []
    return []


def _local_file_storage_paths(tasks: list[dict[str, Any]]) -> list[str]:
    storage_paths: set[str] = set()
    flat_files: list[str] = []
    for task in tasks:
        relative = task.get('data', {}).get('source_relative_path')
        if not isinstance(relative, str):
            continue
        parts = Path(relative).parts
        if len(parts) < 2:
            flat_files.append(relative)
            continue
        storage_paths.add(f'{CONTAINER_LOCAL_FILES_ROOT}/{parts[0]}')
    if flat_files:
        examples = ', '.join(flat_files[:3])
        raise BootstrapError(
            'Label Studio local-file serving requires images to be inside at least one subfolder '
            f'under --images. Move flat image files into a subject/batch folder first. Examples: {examples}'
        )
    return sorted(storage_paths)


def _write_task_manifest(plan: BootstrapPlan, tasks: list[dict[str, Any]]) -> None:
    plan.imports_dir.mkdir(parents=True, exist_ok=True)
    plan.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    plan.task_manifest_path.write_text(json.dumps(tasks, indent=2), encoding='utf-8')


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def _safe_slug(value: str) -> str:
    slug = ''.join(char.lower() if char.isalnum() else '-' for char in value).strip('-')
    return '-'.join(part for part in slug.split('-') if part) or 'labelstudio-project'


def _port_from_url(url: str) -> int:
    return int(url.rsplit(':', 1)[-1])
