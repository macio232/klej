import configparser
import os
import re
import shlex
import uuid
import yaml

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESERVED_SETTINGS = {
    "array",
    "clusters",
    "dependency",
    "error",
    "job-name",
    "hold",
    "kill-on-invalid-dep",
    "output",
}
TOP_LEVEL_KEYS = {
    "profiles",
    "stages",
    "commit",
    "klej",
    "environment",
    "wrapper",
}
KLEJ_CONFIG_KEYS = {"locking": {"mute_warning"}}


@dataclass(frozen=True)
class Profile:
    settings: dict[str, Any]
    environment: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    profiles: dict[str, Profile]
    stages: dict[str, tuple[str, ...]]
    commit: str
    klej: str
    environment: tuple[str, ...]
    wrapper: tuple[str, ...]


def _read_klej_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    if path.exists():
        try:
            with path.open() as file:
                config.read_file(file)
        except (OSError, UnicodeError, configparser.Error) as error:
            raise ValueError(
                f"Invalid klej configuration at {path}: {error}"
            ) from error
    unknown_sections = set(config.sections()) - KLEJ_CONFIG_KEYS.keys()
    if unknown_sections:
        raise ValueError(
            f"{path}: unknown sections: {', '.join(sorted(unknown_sections))}"
        )
    if config.defaults():
        raise ValueError(f"{path}: default options are not supported")
    for section, supported in KLEJ_CONFIG_KEYS.items():
        if section not in config:
            continue
        unknown = set(config[section]) - supported
        if unknown:
            raise ValueError(
                f"{path}: unknown {section} options: "
                + ", ".join(sorted(unknown))
            )
    if config.has_option("locking", "mute_warning"):
        try:
            config.getboolean("locking", "mute_warning")
        except ValueError as error:
            raise ValueError(
                f"{path}: locking.mute_warning must be a boolean"
            ) from error
    return config


def locking_warning_muted(repo_root: Path) -> bool:
    muted = False
    root = repo_root / ".klej"
    for name in ("config", "config.local"):
        config = _read_klej_config(root / name)
        if config.has_option("locking", "mute_warning"):
            muted = config.getboolean("locking", "mute_warning")
    return muted


def mute_locking_warning(repo_root: Path, *, local: bool) -> Path:
    root = repo_root / ".klej"
    path = root / ("config.local" if local else "config")
    config = _read_klej_config(path)
    if "locking" not in config:
        config.add_section("locking")
    config.set("locking", "mute_warning", "true")
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as file:
            config.write(file)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _resolve(value: object, context: str) -> str | int | float | bool:
    if value is None or not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{context} must be a non-null scalar")
    if isinstance(value, str) and value.startswith("$"):
        variable = (
            value[2:-1]
            if value.startswith("${") and value.endswith("}")
            else value[1:]
        )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError(
                f"{context} has invalid environment reference {value!r}"
            )
        if variable not in os.environ:
            raise ValueError(
                f"Environment variable {variable!r} required by {context} is unset"
            )
        return os.environ[variable]
    return value


def _environment(value: object, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        if not all(isinstance(line, str) for line in value):
            raise ValueError(f"{context} list must contain strings")
        return tuple(value)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping, string, or list")
    lines = []
    for name, setting in value.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError(f"{context} contains invalid variable {name!r}")
        if setting is None:
            lines.append(f"unset {name}")
            continue
        values = setting if isinstance(setting, list) else [setting]
        parts = []
        for item in values:
            resolved = _resolve(item, f"{context}.{name}")
            if isinstance(item, str) and item.startswith("$"):
                variable = (
                    item[2:-1]
                    if item.startswith("${") and item.endswith("}")
                    else item[1:]
                )
                parts.append(f"${{{variable}}}")
            else:
                parts.append(shlex.quote(str(resolved)))
        lines.append(f"export {name}={os.pathsep.join(parts)}")
    return tuple(lines)


def _profile(name: str, value: object) -> Profile:
    if not isinstance(value, dict):
        raise TypeError(f"Profile {name!r} must be a mapping")
    values = value.copy()
    environment = _environment(
        values.pop("environment", None), f"profiles.{name}.environment"
    )
    settings: dict[str, Any] = {}
    for key, setting in values.items():
        if not isinstance(key, str):
            raise TypeError(f"Profile {name!r} contains a non-string option")
        option = key.replace("_", "-")
        if option in RESERVED_SETTINGS:
            raise ValueError(
                f"Profile {name!r} cannot override managed option {option!r}"
            )
        resolved = _resolve(setting, f"profiles.{name}.{key}")
        if resolved is not False:
            settings[key] = resolved
    return Profile(settings, environment)


def load_config(path: Path) -> Config:
    return parse_config(_read_document(path), path)


def _read_document(path: Path) -> dict[str, object]:
    try:
        document = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Slurm configuration not found: {path}"
        ) from None
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a mapping")
    return document


def parse_config(document: dict[str, object], path: Path) -> Config:
    if "finalize" in document:
        raise ValueError("finalize has been renamed to klej")
    unknown = set(document) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown keys: {', '.join(sorted(map(str, unknown)))}"
        )
    profiles_value = document.get("profiles")
    stages_value = document.get("stages")
    if not isinstance(profiles_value, dict) or not profiles_value:
        raise ValueError(f"{path}: 'profiles' must be a non-empty mapping")
    if not isinstance(stages_value, dict) or not stages_value:
        raise ValueError(f"{path}: 'stages' must be a non-empty mapping")
    profiles = {
        str(name): _profile(str(name), value)
        for name, value in profiles_value.items()
    }
    stages: dict[str, tuple[str, ...]] = {}
    for name, value in stages_value.items():
        context = f"{path}: stage {name!r}"
        if isinstance(value, str):
            alternatives = (value,)
        elif isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            if not value:
                raise ValueError(f"{context} must name at least one profile")
            alternatives = tuple(value)
        else:
            raise TypeError(
                f"{context} must be a profile name or a list of profile names"
            )
        duplicate = next(
            (
                profile
                for index, profile in enumerate(alternatives)
                if profile in alternatives[:index]
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(
                f"{context} contains duplicate profile {duplicate!r}"
            )
        stages[str(name)] = alternatives
    commit = document.get("commit")
    klej = document.get("klej")
    if not isinstance(commit, str):
        raise TypeError(f"{path}: 'commit' must name a profile")
    if not isinstance(klej, str):
        raise TypeError(f"{path}: 'klej' must name a profile")
    stage_profiles = {
        profile for alternatives in stages.values() for profile in alternatives
    }
    missing = (stage_profiles | {commit, klej}) - profiles.keys()
    if missing:
        raise ValueError(
            f"{path}: undefined profiles: {', '.join(sorted(missing))}"
        )
    wrapper_value = document.get("wrapper", [])
    if not isinstance(wrapper_value, list) or not all(
        isinstance(item, str) for item in wrapper_value
    ):
        raise ValueError(f"{path}: 'wrapper' must be a list of strings")
    if "wrapper" in document and not wrapper_value:
        raise ValueError(f"{path}: 'wrapper' cannot be empty")
    return Config(
        profiles,
        stages,
        commit,
        klej,
        _environment(document.get("environment"), "environment"),
        tuple(wrapper_value),
    )


def _merge_mapping(
    base: dict[object, object], overlay: dict[object, object]
) -> dict[object, object]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_config_files(base: Path | None, overlay: Path) -> Config:
    """Load a pipeline config, optionally overlaying a repository config."""
    if base is None or base.resolve() == overlay.resolve():
        return load_config(overlay)
    document = deepcopy(_read_document(base))
    local = _read_document(overlay)
    for key, value in local.items():
        if key not in {"profiles", "stages"}:
            document[key] = deepcopy(value)
            continue
        current = document.get(key)
        if not isinstance(current, dict) or not isinstance(value, dict):
            document[key] = deepcopy(value)
            continue
        merged = deepcopy(current)
        for name, setting in value.items():
            if (
                key == "profiles"
                and isinstance(merged.get(name), dict)
                and isinstance(setting, dict)
            ):
                merged[name] = _merge_mapping(merged[name], setting)
            else:
                merged[name] = deepcopy(setting)
        document[key] = merged
    return parse_config(document, overlay)


def find_config(
    pipeline: Path, invocation_root: Path, explicit: Path | None = None
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        pipeline.parent / "klej.yaml",
        invocation_root / "klej.yaml",
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    searched = ", ".join(map(str, candidates))
    raise FileNotFoundError(f"No klej.yaml found; searched: {searched}")
