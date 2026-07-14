"""Compatibility shim — real check lives in scripts/check_ckpt.py."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "check_ckpt.py"), run_name="__main__")
