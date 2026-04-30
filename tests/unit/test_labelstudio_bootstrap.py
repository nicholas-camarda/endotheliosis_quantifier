from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace

import pytest

from eq.labelstudio.bootstrap import (
    CONTAINER_LOCAL_FILES_ROOT,
    BootstrapError,
    LabelStudioApiClient,
    build_labelstudio_tasks,
    discover_image_files,
    docker_run_command,
    ensure_docker_available,
    plan_bootstrap,
    run_bootstrap,
    start_labelstudio_container,
)


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'image')
    return path


def test_discovers_supported_images_recursively_and_sorts(tmp_path: Path):
    images = tmp_path / 'images'
    tif = _write_image(images / 'animal_2' / 'b.tif')
    jpg = _write_image(images / 'animal_1' / 'a.JPG')
    _write_image(images / 'notes.txt')

    discovered = discover_image_files(images)

    assert discovered == [jpg, tif]


def test_task_payload_preserves_relative_path_and_subject_hint(tmp_path: Path):
    images = tmp_path / 'images'
    nested = _write_image(images / 'animal_1' / 'kidney_a' / 'image_001.tif')
    flat = _write_image(images / 'flat_image.png')

    tasks = build_labelstudio_tasks(images, [nested, flat])

    assert tasks[0]['data']['image'] == (
        '/data/local-files/?d=animal_1/kidney_a/image_001.tif'
    )
    assert tasks[0]['data']['source_relative_path'] == (
        'animal_1/kidney_a/image_001.tif'
    )
    assert tasks[0]['data']['source_filename'] == 'image_001.tif'
    assert tasks[0]['data']['subject_hint'] == 'animal_1'
    assert tasks[1]['data']['subject_hint'] == 'flat_image'


def test_plan_bootstrap_uses_runtime_labelstudio_root(tmp_path: Path, monkeypatch):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    runtime = tmp_path / 'runtime'
    monkeypatch.setenv('EQ_RUNTIME_ROOT', str(runtime))

    plan = plan_bootstrap(images_dir=images)

    assert plan.runtime_root == runtime / 'labelstudio'
    assert plan.data_dir == runtime / 'labelstudio' / 'data'
    assert plan.media_dir == images
    assert plan.imports_dir == runtime / 'labelstudio' / 'imports'
    assert plan.bootstrap_dir == runtime / 'labelstudio' / 'bootstrap'
    assert plan.project_title == 'EQ Glomerulus Grading'
    assert plan.url == 'http://localhost:8080'
    assert plan.task_count == 1


def test_docker_run_mounts_images_as_local_files_document_root(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')

    command = docker_run_command(plan)

    assert f'{plan.media_dir}:{CONTAINER_LOCAL_FILES_ROOT}:ro' in command
    assert 'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=' + CONTAINER_LOCAL_FILES_ROOT in command


def test_start_container_reuses_existing_container_with_media_mount(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')
    calls = []

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'inspect', plan.container_name]:
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Mounts":[{"Destination":"/label-studio/media"}]}]',
            )
        return SimpleNamespace(returncode=0, stdout='')

    start_labelstudio_container(plan, fake_runner)

    assert ['docker', 'start', plan.container_name] in calls
    assert ['docker', 'rm', '-f', plan.container_name] not in calls


def test_start_container_recreates_failed_nested_media_mount(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')
    calls = []

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'inspect', plan.container_name]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '[{"Mounts":['
                    '{"Destination":"/label-studio/media"},'
                    '{"Destination":"/label-studio/media/images"}'
                    ']}]'
                ),
            )
        return SimpleNamespace(returncode=0, stdout='')

    start_labelstudio_container(plan, fake_runner)

    assert ['docker', 'rm', '-f', plan.container_name] in calls
    assert docker_run_command(plan) in calls


def test_dry_run_writes_manifest_without_docker_or_api(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'

    result = run_bootstrap(
        images_dir=images,
        runtime_root=runtime,
        dry_run=True,
        docker_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('dry-run must not call docker')
        ),
        api_client=SimpleNamespace(
            bootstrap_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('dry-run must not call api')
            )
        ),
    )

    assert result.plan.task_count == 1
    assert result.project_url == ''
    assert result.task_manifest_path.exists()
    assert 'Dry run' in result.message


def test_missing_image_directory_fails_before_side_effects(tmp_path: Path):
    with pytest.raises(BootstrapError, match='Image directory does not exist'):
        plan_bootstrap(images_dir=tmp_path / 'missing')


def test_docker_preflight_starts_installed_docker_desktop_on_macos(monkeypatch):
    calls = []
    attempts = {'docker_info': 0}

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'info']:
            attempts['docker_info'] += 1
            return SimpleNamespace(returncode=1 if attempts['docker_info'] == 1 else 0)
        if command == ['open', '-a', 'Docker']:
            return SimpleNamespace(returncode=0)
        raise AssertionError(f'unexpected command: {command}')

    monkeypatch.setattr('eq.labelstudio.bootstrap.shutil.which', lambda name: '/usr/local/bin/docker')
    monkeypatch.setattr('eq.labelstudio.bootstrap.platform.system', lambda: 'Darwin')
    monkeypatch.setattr(
        'eq.labelstudio.bootstrap.DOCKER_DESKTOP_APP_PATH',
        Path('/Applications/Docker.app'),
    )
    monkeypatch.setattr(Path, 'exists', lambda self: str(self) == '/Applications/Docker.app')
    monkeypatch.setattr('eq.labelstudio.bootstrap.time.sleep', lambda _seconds: None)

    ensure_docker_available(fake_runner, timeout_seconds=2)

    assert ['open', '-a', 'Docker'] in calls
    assert attempts['docker_info'] == 2


def test_docker_preflight_reports_install_command_when_docker_missing_on_macos(monkeypatch):
    monkeypatch.setattr('eq.labelstudio.bootstrap.shutil.which', lambda name: None)
    monkeypatch.setattr('eq.labelstudio.bootstrap.platform.system', lambda: 'Darwin')

    with pytest.raises(BootstrapError, match='brew install --cask docker'):
        ensure_docker_available(lambda command: SimpleNamespace(returncode=1))


def test_labelstudio_readiness_retries_remote_disconnect(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = {'count': 0}

    def flaky_request(method, path, payload=None):
        calls['count'] += 1
        if calls['count'] == 1:
            raise RemoteDisconnected('Remote end closed connection without response')
        return []

    monkeypatch.setattr(client, '_request_json', flaky_request)
    monkeypatch.setattr('eq.labelstudio.bootstrap.time.sleep', lambda _seconds: None)

    client.wait_until_ready(timeout_seconds=2)

    assert calls['count'] == 2


def test_labelstudio_bootstrap_creates_local_file_storages_before_import(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = []

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        if method == 'GET' and path == '/api/projects/':
            return []
        if method == 'POST' and path == '/api/projects/':
            return {'id': 7}
        if method == 'GET' and path == '/api/storages/localfiles/?project=7':
            return []
        if method == 'POST' and path == '/api/storages/localfiles/':
            return {'id': 3}
        if method == 'POST' and path == '/api/projects/7/import':
            return {'task_count': 1}
        raise AssertionError(f'unexpected request: {method} {path} {payload}')

    monkeypatch.setattr(client, '_request_json', fake_request)

    project_id = client.bootstrap_project(
        project_title='Demo',
        label_config='<View />',
        tasks=[
            {
                'data': {
                    'image': '/data/local-files/?d=animal_1/a.png',
                    'source_relative_path': 'animal_1/a.png',
                }
            }
        ],
        timeout_seconds=2,
    )

    assert project_id == 7
    assert calls.index(
        (
            'POST',
            '/api/storages/localfiles/',
            {
                'project': 7,
                'path': f'{CONTAINER_LOCAL_FILES_ROOT}/animal_1',
                'title': 'EQ Image Media animal_1',
                'use_blob_urls': True,
                'recursive_scan': True,
            },
        )
    ) < calls.index(
        (
            'POST',
            '/api/projects/7/import',
            [
                {
                    'data': {
                        'image': '/data/local-files/?d=animal_1/a.png',
                        'source_relative_path': 'animal_1/a.png',
                    }
                }
            ],
        )
    )


def test_labelstudio_cli_command_forwards_start_options(monkeypatch, tmp_path: Path, capsys):
    import eq.__main__ as main

    captured = {}

    def fake_run_bootstrap(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message='Dry run: ready',
            project_url='',
            task_manifest_path=tmp_path / 'tasks.json',
            plan=SimpleNamespace(url='http://localhost:9090'),
        )

    monkeypatch.setattr(main, '_load_labelstudio_bootstrap', lambda: fake_run_bootstrap)

    args = SimpleNamespace(
        images=str(tmp_path / 'images'),
        project_name='My Project',
        runtime_root=str(tmp_path / 'runtime'),
        port=9090,
        container_name='custom-ls',
        docker_image='labelstudio:test',
        username='admin@example.test',
        password='password',
        api_token='token123',
        timeout_seconds=5,
        dry_run=True,
    )

    main.labelstudio_start_command(args)

    output = capsys.readouterr().out
    assert captured['images_dir'] == tmp_path / 'images'
    assert captured['project_name'] == 'My Project'
    assert captured['runtime_root'] == tmp_path / 'runtime'
    assert captured['port'] == 9090
    assert captured['container_name'] == 'custom-ls'
    assert captured['docker_image'] == 'labelstudio:test'
    assert captured['username'] == 'admin@example.test'
    assert captured['password'] == 'password'
    assert captured['api_token'] == 'token123'
    assert captured['timeout_seconds'] == 5
    assert captured['dry_run'] is True
    assert 'Dry run: ready' in output
    assert 'Login email: admin@example.test' in output
    assert 'Login password: password' in output
