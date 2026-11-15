from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    total = 0
    for test_file in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            getattr(module, name)()
            print(f"PASS {test_file.name}::{name}")
            total += 1
    print(f"{total} CPU checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
