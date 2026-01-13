from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config():
    if not GlobalHydra.instance().is_initialized():
        with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
            return compose(config_name="config")
    return compose(config_name="config")
