from pathlib import Path

from forgeff.evaluate.setting import load_setting_evaluate
from forgeff.grade.setting import load_setting_grade
from forgeff.train.setting import load_setting_train


def _write_setting(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_train_setting_paths_are_resolved_relative_to_file(tmp_path: Path) -> None:
    setting_file = _write_setting(
        tmp_path,
        "forgeff.train.toml",
        """
[configurations]
training = ["data/training.cfg"]

[potentials]
initial = "potentials/initial.toml"
final = "potentials/final.toml"
""",
    )

    setting = load_setting_train(setting_file)
    assert setting.configurations.training == [str((tmp_path / "data/training.cfg").resolve())]
    assert setting.potentials.initial == str((tmp_path / "potentials/initial.toml").resolve())
    assert setting.potentials.final == str((tmp_path / "potentials/final.toml").resolve())


def test_evaluate_setting_paths_are_resolved_relative_to_file(tmp_path: Path) -> None:
    setting_file = _write_setting(
        tmp_path,
        "forgeff.evaluate.toml",
        """
[configurations]
initial = ["data/initial.cfg"]
final = ["data/final.cfg"]

[potentials]
final = "potentials/final.toml"
""",
    )

    setting = load_setting_evaluate(setting_file)
    assert setting.configurations.initial == [str((tmp_path / "data/initial.cfg").resolve())]
    assert setting.configurations.final == [str((tmp_path / "data/final.cfg").resolve())]
    assert setting.potentials.final == str((tmp_path / "potentials/final.toml").resolve())


def test_grade_setting_paths_are_resolved_relative_to_file(tmp_path: Path) -> None:
    setting_file = _write_setting(
        tmp_path,
        "forgeff.grade.toml",
        """
[configurations]
training = ["data/training.cfg"]
initial = ["data/initial.cfg"]
final = ["data/final.cfg"]

[potentials]
final = "potentials/final.toml"
""",
    )

    setting = load_setting_grade(setting_file)
    assert setting.configurations.training == [str((tmp_path / "data/training.cfg").resolve())]
    assert setting.configurations.initial == [str((tmp_path / "data/initial.cfg").resolve())]
    assert setting.configurations.final == [str((tmp_path / "data/final.cfg").resolve())]
    assert setting.potentials.final == str((tmp_path / "potentials/final.toml").resolve())
