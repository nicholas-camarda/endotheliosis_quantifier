from pathlib import Path
from types import SimpleNamespace

import pytest

from eq.labelstudio.bootstrap import (
    BootstrapError,
    build_labelstudio_tasks,
    discover_image_files,
    plan_bootstrap,
    run_bootstrap,
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
    assert 'Dry run: ready' in capsys.readouterr().out
