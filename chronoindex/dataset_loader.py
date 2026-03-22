from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import os
from pathlib import Path
import tomllib
from typing import Any, Iterable


DEFAULT_CONFIG_PATH = Path("datasets.toml")
_PROJECT_ROOT_MARKERS = ("setup.py", "pyproject.toml", ".git")


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Resolve the repository root by looking for standard project markers."""
    current = Path(start or Path.cwd()).resolve()
    search_roots = [current, *current.parents]
    for root in search_roots:
        for marker in _PROJECT_ROOT_MARKERS:
            if (root / marker).exists():
                return root
    return current


def _resolve_config_path(config_path: str | Path | None) -> tuple[Path, Path]:
    project_root = resolve_project_root()
    raw_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    resolved = raw_path if raw_path.is_absolute() else project_root / raw_path
    return project_root, resolved


def _load_config(config_path: str | Path | None) -> tuple[Path, Path, dict[str, Any]]:
    project_root, resolved_config_path = _resolve_config_path(config_path)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {resolved_config_path}")

    with resolved_config_path.open("rb") as fp:
        raw = tomllib.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid dataset config format in {resolved_config_path}")

    return project_root, resolved_config_path, raw


def _handler_class_from_spec(handler_spec: str):
    if ":" not in handler_spec:
        raise ValueError(
            "Handler must be in 'module.path:ClassName' format, "
            "e.g. 'chronoindex.dataset_ingestion.zhao_cgm_dataset:ZhaoCGMDataset'."
        )
    module_path, class_name = handler_spec.split(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise ValueError(f"Class '{class_name}' not found in module '{module_path}'.")
    return getattr(module, class_name)


def _should_treat_as_path(key: str, explicit_path_keys: set[str]) -> bool:
    if key in explicit_path_keys:
        return True
    lowered = key.lower()
    path_tokens = ("path", "folder", "file", "dir")
    return any(token in lowered for token in path_tokens)


def _resolve_path_value(value: Any, project_root: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if "://" in value:
            return value
        expanded = Path(os.path.expandvars(value)).expanduser()
        if expanded.is_absolute():
            return str(expanded)
        return str((project_root / expanded).resolve())
    if isinstance(value, list):
        return [_resolve_path_value(item, project_root) for item in value]
    return value


def _resolve_kwargs_paths(
    kwargs: dict[str, Any], project_root: Path, explicit_path_keys: set[str]
) -> dict[str, Any]:
    resolved = dict(kwargs)
    for key, value in kwargs.items():
        if _should_treat_as_path(key, explicit_path_keys):
            resolved[key] = _resolve_path_value(value, project_root)
    return resolved


def list_dataset_entries(
    config_path: str | Path | None = None,
    only: Iterable[str] | None = None,
    include_disabled: bool = False,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """
    Parse dataset entries from TOML config and resolve path kwargs to absolute paths.

    Returns:
      (config_file_path, entries_dict)
    """
    project_root, resolved_config_path, raw = _load_config(config_path)
    datasets_section = raw.get("datasets", {})
    if not isinstance(datasets_section, dict):
        raise ValueError(
            f"Invalid config: 'datasets' table missing or not a table in {resolved_config_path}"
        )

    allowed = set(only) if only else None
    entries: dict[str, dict[str, Any]] = {}

    for dataset_name, entry in datasets_section.items():
        if allowed is not None and dataset_name not in allowed:
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid dataset entry for '{dataset_name}' in {resolved_config_path}"
            )

        enabled = bool(entry.get("enabled", True))
        if not enabled and not include_disabled:
            continue

        handler = entry.get("handler")
        if not isinstance(handler, str) or not handler.strip():
            raise ValueError(
                f"Dataset '{dataset_name}' is missing a valid 'handler' string."
            )

        harmonizer = entry.get("harmonizer")
        if harmonizer is not None:
            if not isinstance(harmonizer, str) or not harmonizer.strip() or ":" not in harmonizer:
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid 'harmonizer'. "
                    "Expected 'module.path:function_name'."
                )

        kwargs = entry.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"Dataset '{dataset_name}' has non-table 'kwargs'. Expected [datasets.{dataset_name}.kwargs]."
            )

        raw_path_keys = entry.get("path_kwargs", [])
        if raw_path_keys is None:
            raw_path_keys = []
        if not isinstance(raw_path_keys, list):
            raise ValueError(
                f"Dataset '{dataset_name}' has invalid 'path_kwargs'. Expected a list."
            )
        explicit_path_keys = set(raw_path_keys)
        resolved_kwargs = _resolve_kwargs_paths(kwargs, project_root, explicit_path_keys)

        entries[dataset_name] = {
            "handler": handler,
            "harmonizer": harmonizer,
            "enabled": enabled,
            "kwargs": resolved_kwargs,
            "path_kwargs": sorted(explicit_path_keys),
        }

    if allowed is not None:
        unknown = sorted(allowed.difference(datasets_section.keys()))
        if unknown:
            raise ValueError(
                f"Unknown dataset names in --only: {', '.join(unknown)}"
            )

    return resolved_config_path, entries


def load_datasets(
    config_path: str | Path | None = None,
    only: Iterable[str] | None = None,
    include_disabled: bool = False,
    fail_fast: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Load datasets configured in TOML.

    Returns:
      loaded: {dataset_name: dataset_instance}
      errors: {dataset_name: error_message}
    """
    _, entries = list_dataset_entries(
        config_path=config_path,
        only=only,
        include_disabled=include_disabled,
    )
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _load_one(dataset_name: str, entry: dict[str, Any]):
        print(f"➡️ Loading datasets.{dataset_name}")
        handler_cls = _handler_class_from_spec(entry["handler"])
        return handler_cls(**entry["kwargs"])

    if parallel and len(entries) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_one, dataset_name, entry): dataset_name
                for dataset_name, entry in entries.items()
            }
            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    loaded[dataset_name] = future.result()
                except Exception as exc:
                    errors[dataset_name] = str(exc)
                    if fail_fast:
                        raise
    else:
        for dataset_name, entry in entries.items():
            try:
                loaded[dataset_name] = _load_one(dataset_name, entry)
            except Exception as exc:
                errors[dataset_name] = str(exc)
                if fail_fast:
                    raise

    return loaded, errors
