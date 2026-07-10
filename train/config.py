from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def flatten_sections(cfg: SimpleNamespace, *sections: str) -> SimpleNamespace:
    merged = {}
    for name in sections:
        section = getattr(cfg, name)
        merged.update(vars(section))
    return SimpleNamespace(**merged)
