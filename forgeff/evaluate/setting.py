from dataclasses import dataclass, field
from pathlib import Path

from forgeff.setting import (
    CommonSetting,
    ConfigurationsBase,
    DataclassFromAny,
    parse_setting,
    resolve_setting_paths,
    resolve_setting_path,
)


@dataclass
class _Configurations(ConfigurationsBase):
    """Configurations."""

    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class _Potentials(DataclassFromAny):
    """Potentials."""

    final: str = "final.npy"


@dataclass
class _Setting(DataclassFromAny):
    """Setting for the application of the potential."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: _Configurations = field(default_factory=_Configurations)
    potentials: _Potentials = field(default_factory=_Potentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.common = CommonSetting.from_any(self.common)
        self.configurations = _Configurations.from_any(self.configurations)
        self.potentials = _Potentials.from_any(self.potentials)


def load_setting_evaluate(filename: str | Path | None = None) -> _Setting:
    """Load setting for `evaluate`.

    Returns
    -------
    EvaluateSetting

    """
    if filename is None:
        return _Setting()
    setting = _Setting(**parse_setting(filename))
    base = Path(filename).resolve().parent
    setting.configurations.initial = resolve_setting_paths(base, setting.configurations.initial)
    setting.configurations.final = resolve_setting_paths(base, setting.configurations.final)
    setting.potentials.final = resolve_setting_path(base, setting.potentials.final)
    return setting
