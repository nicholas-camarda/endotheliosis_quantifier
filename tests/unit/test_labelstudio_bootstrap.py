import json
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eq.labelstudio.bootstrap import (
    CONTAINER_LOCAL_FILES_ROOT,
    BootstrapError,
    LabelStudioApiClient,
    _prediction_from_mask,
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


def test_prediction_from_mask_splits_disconnected_regions(tmp_path: Path):
    mask_path = tmp_path / 'mask.png'
    array = np.zeros((120, 120), dtype=np.uint8)
    array[5:40, 5:40] = 255
    array[70:105, 70:105] = 255
    Image.fromarray(array, mode='L').save(mask_path)

    prediction = _prediction_from_mask(
        prediction_id='auto_123',
        mask_path=mask_path,
        mask_release_id='release_x',
    )

    assert prediction['model_version'] == 'medsam-release:release_x'
    assert len(prediction['result']) == 2
    assert prediction['result'][0]['id'] == 'auto_123_001'
    assert prediction['result'][1]['id'] == 'auto_123_002'


def test_prediction_from_mask_drops_tiny_components(tmp_path: Path):
    mask_path = tmp_path / 'mask_tiny.png'
    array = np.zeros((80, 80), dtype=np.uint8)
    array[10:40, 10:40] = 255
    array[60:63, 60:63] = 255
    Image.fromarray(array, mode='L').save(mask_path)

    prediction = _prediction_from_mask(
        prediction_id='auto_200',
        mask_path=mask_path,
        mask_release_id='release_y',
    )

    assert len(prediction['result']) == 1
    assert prediction['result'][0]['id'] == 'auto_200_001'


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
    assert result.companion_status == 'hybrid-enabled companion-required'


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


def test_labelstudio_readiness_retries_connection_reset(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = {'count': 0}

    def flaky_request(method, path, payload=None):
        calls['count'] += 1
        if calls['count'] == 1:
            raise ConnectionResetError('connection reset by peer')
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
        if method == 'GET' and path == '/api/tasks/?project=7':
            return {
                'tasks': [
                    {
                        'id': 11,
                        'total_annotations': 0,
                        'data': {'source_relative_path': 'animal_1/a.png'},
                    }
                ]
            }
        if method == 'POST' and path == '/api/tasks/11/annotations/':
            return {'id': 99}
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
                },
                'predictions': [
                    {
                        'result': [
                            {
                                'id': 'auto_001',
                                'from_name': 'glomerulus_roi',
                                'to_name': 'image',
                                'type': 'brushlabels',
                                'value': {
                                    'format': 'rle',
                                    'rle': [0, 1],
                                    'brushlabels': ['complete_glomerulus'],
                                },
                            }
                        ]
                    }
                ],
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
                    },
                    'predictions': [
                        {
                            'result': [
                                {
                                    'id': 'auto_001',
                                    'from_name': 'glomerulus_roi',
                                    'to_name': 'image',
                                    'type': 'brushlabels',
                                    'value': {
                                        'format': 'rle',
                                        'rle': [0, 1],
                                        'brushlabels': ['complete_glomerulus'],
                                    },
                                }
                            ]
                        }
                    ],
                }
            ],
        )
    )
    assert (
        'POST',
        '/api/tasks/11/annotations/',
        {
            'result': [
                {
                    'id': 'auto_001',
                    'from_name': 'glomerulus_roi',
                    'to_name': 'image',
                    'type': 'brushlabels',
                    'value': {
                        'format': 'rle',
                        'rle': [0, 1],
                        'brushlabels': ['complete_glomerulus'],
                    },
                }
            ]
        },
    ) in calls


def test_labelstudio_iter_project_tasks_supports_tasks_key(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')

    def fake_request(method, path, payload=None):
        assert method == 'GET'
        assert path == '/api/tasks/?project=7'
        return {'total': 1, 'tasks': [{'id': 17}]}

    monkeypatch.setattr(client, '_request_json', fake_request)
    tasks = list(client._iter_project_tasks(7))
    assert tasks == [{'id': 17}]


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
            companion_status='hybrid-disabled',
        )

    monkeypatch.setattr(main, '_load_labelstudio_bootstrap', lambda: fake_run_bootstrap)

    args = SimpleNamespace(
        image_dir=None,
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
        config=str(tmp_path / 'cfg.yaml'),
        hybrid_mode='auto',
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
    assert captured['config_path'] == tmp_path / 'cfg.yaml'
    assert captured['hybrid_mode'] == 'auto'
    assert 'Dry run: ready' in output
    assert 'Login email: admin@example.test' in output
    assert 'Login password: password' in output


def test_labelstudio_cli_accepts_positional_image_dir(monkeypatch, tmp_path: Path):
    import eq.__main__ as main

    captured = {}

    def fake_run_bootstrap(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message='ready',
            project_url='',
            task_manifest_path=tmp_path / 'tasks.json',
            plan=SimpleNamespace(url='http://localhost:9090'),
            companion_status='hybrid-disabled',
        )

    monkeypatch.setattr(main, '_load_labelstudio_bootstrap', lambda: fake_run_bootstrap)
    args = SimpleNamespace(
        image_dir=str(tmp_path / 'images'),
        images=None,
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
        config=None,
        hybrid_mode='disabled',
    )

    main.labelstudio_start_command(args)
    assert captured['images_dir'] == tmp_path / 'images'


def test_hybrid_dry_run_includes_preload_predictions(tmp_path: Path):
    images = tmp_path / 'images'
    img = _write_image(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'
    mask_dir = (
        runtime
        / 'derived_data'
        / 'generated_masks'
        / 'glomeruli'
        / 'medsam_finetuned'
        / 'release_a'
        / 'masks'
    )
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / 'mask_a.png'
    Image.new('L', (2, 2), color=255).save(mask_path)
    manifest_dir = runtime / 'derived_data' / 'generated_masks' / 'glomeruli'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / 'manifest.csv').write_text(
        '\n'.join(
            [
                'generated_mask_id,mask_release_id,mask_source,adoption_tier,cohort_id,lane_assignment,source_sample_id,source_image_path,reference_mask_path,generated_mask_path,release_manifest_path,checkpoint_id,checkpoint_path,proposal_source,proposal_threshold,run_id,generation_status,provenance_path',
                f'row1,release_a,medsam_finetuned_glomeruli,tier,cohort,lane,sample,{img},,{mask_path},{mask_dir.parent / "manifest.csv"},ckpt,,auto,0.2,run,generated,prov.json',
            ]
        ),
        encoding='utf-8',
    )
    (mask_dir.parent / 'manifest.csv').write_text(
        '\n'.join(
            [
                'generated_mask_id,mask_release_id,source_image_path,generated_mask_path,generation_status',
                f'row1,release_a,{img},{mask_path},generated',
            ]
        ),
        encoding='utf-8',
    )
    cfg = tmp_path / 'hybrid.yaml'
    cfg.write_text(
        'hybrid:\n  enabled: true\n  selection_mode: latest_valid\n  require_box_assisted_medsam: false\ncompanion:\n  base_url: "http://localhost:8098"\n  health_path: "/healthz"\n',
        encoding='utf-8',
    )

    result = run_bootstrap(
        images_dir=images,
        runtime_root=runtime,
        dry_run=True,
        config_path=cfg,
        hybrid_mode='enabled',
        docker_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('dry-run must not call docker')
        ),
        api_client=SimpleNamespace(
            bootstrap_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('dry-run must not call api')
            )
        ),
    )
    tasks = json.loads(result.task_manifest_path.read_text(encoding='utf-8'))
    assert tasks[0]['data']['preload_prediction_status'] == 'available'
    assert 'predictions' in tasks[0]
