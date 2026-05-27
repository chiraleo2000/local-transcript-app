"""Path resolution for dev and PyInstaller installs."""

from pathlib import Path

from backend.paths import app_root, is_frozen, resolve_path


def test_app_root_is_directory():
    root = app_root()
    assert root.is_dir()


def test_resolve_relative_models_path():
    path = resolve_path("./models")
    assert path.name == "models"
    assert path.is_absolute()


def test_is_frozen_false_in_dev():
    assert is_frozen() is False
