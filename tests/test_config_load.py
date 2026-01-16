from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_config_load():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir), job_name="test"):
        cfg = compose(config_name="config")

    assert cfg.seed is not None
    assert "training" in cfg
    assert "model" in cfg
