from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bsg.experiments.main_config import config_hash, resolve_main_config


path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "main_experiment.yaml"
for tier in ("smoke", "core", "full"):
    config = resolve_main_config(path, tier)
    print(tier, "ok", config_hash(config), [model["id"] for model in config["selected_models"]])
