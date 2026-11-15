"""Small config loader with a PyYAML fallback for repository configs."""

from __future__ import annotations

from pathlib import Path
import json


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        out = {}
        if inner:
            for part in inner.split(","):
                k, v = part.split(":", 1)
                out[k.strip()] = _parse_scalar(v.strip())
        return out
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _fallback_yaml(text: str):
    tokens = []
    for raw in text.splitlines():
        clean = _strip_comment(raw)
        if clean.strip():
            tokens.append((len(clean) - len(clean.lstrip(" ")), clean.strip()))

    def parse(index: int, indent: int):
        is_list = tokens[index][1].startswith("- ")
        out = [] if is_list else {}
        while index < len(tokens):
            current_indent, content = tokens[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"invalid fallback YAML indentation near {content!r}")
            if is_list:
                if not content.startswith("- "):
                    break
                out.append(_parse_scalar(content[2:]))
                index += 1
                continue
            if content.startswith("- ") or ":" not in content:
                break
            key, value = content.split(":", 1)
            index += 1
            if value.strip():
                out[key.strip()] = _parse_scalar(value)
            elif index < len(tokens) and tokens[index][0] > indent:
                out[key.strip()], index = parse(index, tokens[index][0])
            else:
                out[key.strip()] = {}
        return out, index

    return {} if not tokens else parse(0, tokens[0][0])[0]


def load_config(path: str | Path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _fallback_yaml(text)
