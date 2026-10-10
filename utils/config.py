import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACC = "Patient's Accession Number"
IDX = "Number of Brain CT?"
NEW_SURGERY = "Does this CT have new neurological surgery  ?"
NON_TBI = "Non-TBI agents\n(like tumor, stroke, etc.)"


def load_config(path=None, overrides=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(path.read_text())
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value) if isinstance(value, str) else value
    return cfg


def p(cfg, key):
    """Resolve a config path relative to the project root."""
    return (ROOT / cfg["paths"][key]).resolve()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def parse_overrides(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = v
    return out
