"""Package-local paths for bundled models."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_ROOT / "models"
FRANKA_PANDA_MODEL_DIR = MODELS_DIR / "franka_panda"


def get_franka_model_path(filename: str) -> Path:
    """Return a path inside the bundled Franka Panda MuJoCo model directory."""
    path = FRANKA_PANDA_MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Bundled Franka model file not found: {path}")
    return path
