from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import numpy as np
from ase import Atoms
import pytest

import forgeff.calculator as calculator_module
from forgeff.calculator import make_calculator
from forgeff.energy_offsets import apply_species_energy_offsets
from forgeff.io import read_potential
from forgeff.io import write_potential
from forgeff.io.mlip.cfg import _convert_species, _parse_value
from forgeff.io.potfit import write_force
from forgeff.loss import ErrorPrinter, LossFunction, LossFunctionBase, LossFunctionStress
from forgeff.loss import LossSetting
from forgeff.loss import _resolve_species_energy_offsets
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.ase.engine import GenericASEEngine
from forgeff.potentials.ase.custom import CustomPairPotential
from forgeff.potentials.ase.numba_pair import NumbaPairPotential
import forgeff.potentials.ase.neighbor_cache as neighbor_cache_module
from forgeff.potentials.sw.data import SWData
from forgeff.potentials.sw.numpy import NumpySWEngine
from forgeff.potentials.sw.numba import NumbaSWEngine
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.numpy.adp_engine import NumpyADPEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.optimizers.randomizer import Randomizer
from forgeff.parallel import DummyMPIComm
from forgeff.train.setting import _convert_steps
from forgeff.train.trainer import Trainer, _validate_potential_species_order


def test_parallel_import_falls_back_when_mpi_runtime_is_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def _missing_mpi(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mpi4py":
            raise RuntimeError("cannot load MPI library")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_mpi)

    parallel = importlib.reload(importlib.import_module("forgeff.parallel"))

    assert parallel.world.__class__.__name__ == "DummyMPIComm"
    assert parallel.world.size == 1


@pytest.mark.parametrize(
    "factory_name",
    ["make_eam_engine", "make_adp_engine", "make_tersoff_engine", "make_sw_engine"],
)
def test_numba_factories_fail_cleanly_when_numba_is_missing(monkeypatch, factory_name: str) -> None:
    original_import_module = importlib.import_module

    def _missing_numba(module_path: str):
        if "numba" in module_path:
            raise ModuleNotFoundError("No module named 'numba'")
        return original_import_module(module_path)

    monkeypatch.setattr(importlib, "import_module", _missing_numba)
    factory = getattr(calculator_module, factory_name)

    with pytest.raises(RuntimeError, match="no numba"):
        factory("numba")


@pytest.mark.parametrize(
    "module_name",
    [
        "forgeff.potentials.ase.numba_pair",
        "forgeff.potentials.eam.numba.eam_engine",
        "forgeff.potentials.eam.numba.adp_engine",
        "forgeff.potentials.sw.numba",
        "forgeff.potentials.tersoff.numba",
    ],
)
def test_direct_numba_module_imports_fail_cleanly_when_numba_is_missing(
    monkeypatch, module_name: str
) -> None:
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(RuntimeError, match="no numba"):
        importlib.import_module(module_name)


def test_empty_atoms_stress_loss_skips_non_3d_cells() -> None:
    atoms = Atoms()
    atoms.calc = SimpleNamespace(
        results={"stress": np.zeros(6, dtype=float)},
        targets={"stress": np.zeros(6, dtype=float)},
    )

    pot_data = SimpleNamespace(number_of_parameters_optimized=1)
    loss = LossFunctionStress([atoms], pot_data)

    assert loss.idcs_str.size == 0
    assert loss.calculate() == 0.0


def test_stress_loss_uses_global_volume_indices_for_mixed_cells() -> None:
    cluster = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    cluster.calc = SimpleNamespace(
        results={"stress": np.zeros(6, dtype=float)},
        targets={"stress": np.zeros(6, dtype=float)},
    )
    bulk = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[2.0, 2.0, 2.0], pbc=True)
    bulk.calc = SimpleNamespace(
        results={"stress": np.ones(6, dtype=float)},
        targets={"stress": np.zeros(6, dtype=float)},
    )

    pot_data = SimpleNamespace(number_of_parameters_optimized=1)
    loss = LossFunctionStress([cluster, bulk], pot_data)

    assert loss.idcs_str.tolist() == [1]
    np.testing.assert_allclose(loss.volumes, np.array([0.0, 8.0]))
    assert np.isfinite(loss.calculate())


def test_empty_atoms_error_printer_skips_non_3d_cells() -> None:
    atoms = Atoms()
    atoms.calc = SimpleNamespace(
        results={"energy": 0.0, "stress": np.zeros(6, dtype=float)},
        targets={"energy": 0.0, "stress": np.zeros(6, dtype=float)},
    )

    printer = ErrorPrinter([atoms])

    assert printer.idcs_str.size == 0
    errors = printer.calculate()
    assert errors["stress"]["N"] == 0


def test_loss_jac_updates_parameters_before_differentiating() -> None:
    pot_data = ASEData(
        engine="numpy",
        calculator_kwargs={
            "calculator": "numpy",
            "expression": "epsilon * r",
            "parameter_names": ["epsilon"],
            "cutoff": 3.0,
        },
    )
    pot_data.add_parameter("epsilon", (), 1.0)
    atoms = Atoms("Al2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.calc = SimpleNamespace(results={"energy": 0.0})
    loss = LossFunction(
        [atoms],
        pot_data,
        LossSetting(forces_weight=0.0, stress_weight=0.0, energy_per_atom=False),
        engine="numpy",
    )

    jac = loss.jac(np.array([2.0]))

    np.testing.assert_allclose(loss.pot_data.parameters, np.array([2.0]))
    np.testing.assert_allclose(jac, np.array([4.0]), rtol=1e-8, atol=1e-8)


def test_loss_only_updates_local_images_on_each_mpi_rank() -> None:
    class _Comm:
        rank = 1
        size = 2

        def bcast(self, obj, root=0):
            return obj

        def allreduce(self, obj, op=None):
            return obj

        def Allreduce(self, sendobj, recvobj=None, op=None):
            if recvobj is not None:
                recvobj[...] = sendobj[...]

    class _Calc:
        def __init__(self) -> None:
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((1, 3), dtype=float),
                "stress": np.zeros(6, dtype=float),
            }
            self.targets = {
                "energy": 0.0,
                "forces": np.zeros((1, 3), dtype=float),
                "stress": np.zeros(6, dtype=float),
            }
            self.update_count = 0

        def update_parameters(self, pot_data) -> None:
            self.update_count += 1

    class _Loss(LossFunctionBase):
        def __call__(self, parameters):
            self._set_parameters(parameters)
            return 0.0

    pot_data = SimpleNamespace(parameters=np.array([0.0]), number_of_parameters_optimized=1)
    images = []
    for _ in range(4):
        atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)
        atoms.calc = _Calc()
        images.append(atoms)

    loss = _Loss(images, pot_data, LossSetting(), comm=_Comm())
    loss._set_parameters(np.array([2.0]))

    update_counts = [atoms.calc.update_count for atoms in loss.images]
    assert update_counts == [0, 1, 0, 1]


@pytest.mark.parametrize(
    "engine_factory, kwargs",
    [
        (
            CustomPairPotential,
            {
                "expression": "de * (exp(-2*a*(r-re)) - 2*exp(-a*(r-re)))",
                "parameter_names": ["de", "a", "re"],
                "de": 1.0,
                "a": 1.0,
                "re": 1.0,
                "cutoff": 5.0,
            },
        ),
        (
            NumbaPairPotential,
            {
                "form": "morse",
                "De": 1.0,
                "a": 1.0,
                "re": 1.0,
                "cutoff": 5.0,
            },
        ),
    ],
)
def test_pair_engines_cache_neighbor_lists_for_repeated_calls(
    monkeypatch,
    engine_factory,
    kwargs,
) -> None:
    call_count = {"value": 0}
    original_neighbor_list = neighbor_cache_module.neighbor_list

    def _counting_neighbor_list(*args, **kwargs):
        call_count["value"] += 1
        return original_neighbor_list(*args, **kwargs)

    monkeypatch.setattr(neighbor_cache_module, "neighbor_list", _counting_neighbor_list)

    atoms = Atoms(
        "Al2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.calc = engine_factory(**kwargs)

    atoms.get_potential_energy()
    assert call_count["value"] == 1

    atoms.get_potential_energy()
    assert call_count["value"] == 1

    atoms.positions[1, 0] += 0.1
    atoms.get_potential_energy()
    assert call_count["value"] == 2


def test_parse_value_accepts_scientific_and_signed_floats() -> None:
    assert _parse_value("1.23e-4") == pytest.approx(1.23e-4)
    assert _parse_value("+1.23") == pytest.approx(1.23)
    assert _parse_value("-7") == -7
    assert _parse_value("true") is True


def test_convert_species_accepts_empty_lists() -> None:
    assert _convert_species([]) == []


def test_convert_steps_preserves_kwargs_for_minimize_methods() -> None:
    steps = _convert_steps(["L-BFGS-B", {"method": "CG", "kwargs": {"maxiter": 3}}])

    assert steps[0]["method"] == "minimize"
    assert steps[0]["kwargs"]["method"] == "L-BFGS-B"
    assert steps[1]["method"] == "minimize"
    assert steps[1]["kwargs"]["method"] == "CG"
    assert steps[1]["kwargs"]["maxiter"] == 3


def test_species_order_mismatch_is_rejected_before_training() -> None:
    with pytest.raises(ValueError, match="Training species order does not match"):
        _validate_potential_species_order(["C", "H", "O"], ["Al", "Cu"])


def test_update_mindist_ignores_diagonal_self_distances() -> None:
    pot_data = SimpleNamespace(min_dist=10.0)
    trainer = Trainer(pot_data, comm=DummyMPIComm())
    images = [
        Atoms("Al", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Al2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True),
    ]

    trainer.update_mindist(images)

    assert pot_data.min_dist == pytest.approx(1.5)


def test_species_energy_offsets_preserve_free_energy_when_present() -> None:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    results = {"energy": 1.0, "free_energy": 1.5, "energies": np.array([0.4, 0.6])}
    adjusted = apply_species_energy_offsets(results, atoms, SimpleNamespace(species_energy_offsets={"Si": 2.0}))

    assert adjusted["energy"] == pytest.approx(5.0)
    assert adjusted["free_energy"] == pytest.approx(5.5)
    np.testing.assert_allclose(adjusted["energies"], np.array([2.4, 2.6]))


def test_species_energy_offset_regression_uses_ase_results_energy() -> None:
    images = []
    for symbols, energy in [("CH4", -10.0), ("C2H6O", -20.0)]:
        atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
        atoms.calc = SimpleNamespace(results={"energy": energy})
        images.append(atoms)

    pot_data = SimpleNamespace(species=["C", "H", "O"], species_energy_offsets={})
    setting = LossSetting(species_energy_offset_mode="regression")

    offsets = _resolve_species_energy_offsets(images, pot_data, setting)

    assert set(offsets) == {"C", "H", "O"}
    assert all(np.isfinite(list(offsets.values())))


def test_randomizer_uses_seed_and_attribute_access() -> None:
    class _PotData:
        def __init__(self) -> None:
            self.species_coeffs = np.zeros(2, dtype=float)
            self.radial_coeffs = np.zeros((2, 2), dtype=float)
            self.moment_coeffs = np.zeros(1, dtype=float)
            self.optimized = []

        @property
        def parameters(self) -> np.ndarray:
            return np.concatenate(
                [
                    np.asarray(self.species_coeffs, dtype=float).ravel(),
                    np.asarray(self.radial_coeffs, dtype=float).ravel(),
                    np.asarray(self.moment_coeffs, dtype=float).ravel(),
                ]
            )

    class _Loss:
        def __init__(self) -> None:
            self.comm = DummyMPIComm()
            self.pot_data = _PotData()

        def __call__(self, parameters):
            return float(np.sum(np.asarray(parameters, dtype=float) ** 2))

        def gather_data(self) -> None:
            return None

    loss = _Loss()
    optimizer = Randomizer(loss, optimized=["species_coeffs", "radial_coeffs", "moment_coeffs"])
    params = optimizer._optimize(seed=123)

    assert params.shape == (7,)
    assert not np.allclose(loss.pot_data.species_coeffs, 0.0)
    assert not np.allclose(loss.pot_data.radial_coeffs, 0.0)
    assert not np.allclose(loss.pot_data.moment_coeffs, 0.0)


def test_ase_engine_adapter_jacobians_are_finite_difference_values() -> None:
    pot_data = ASEData(
        engine="numpy",
        calculator_kwargs={
            "calculator": "numpy",
            "expression": "epsilon * r",
            "parameter_names": ["epsilon"],
            "cutoff": 3.0,
        },
    )
    pot_data.add_parameter("epsilon", (), 2.0)
    atoms = Atoms("Al2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    calc = make_calculator(pot_data, engine="numpy")

    np.testing.assert_allclose(calc.engine.jac_energy(atoms).parameters, np.array([1.5]), rtol=1e-8, atol=1e-8)


def test_ase_engine_adapter_handles_empty_atoms() -> None:
    pot_data = ASEData(
        engine="numpy",
        calculator_kwargs={
            "calculator": "LennardJones",
            "epsilon": 1.0,
            "sigma": 1.0,
            "rc": 2.5,
        },
    )
    engine = GenericASEEngine(pot_data)
    result = engine.calculate(Atoms())

    assert result["energy"] == 0.0
    assert result["energies"].shape == (0,)
    assert result["forces"].shape == (0, 3)


def test_ase_engine_adapter_uses_pairwise_site_energies() -> None:
    pot_data = ASEData(
        engine="numpy",
        calculator_kwargs={
            "calculator": "LennardJones",
            "epsilon": 1.0,
            "sigma": 1.0,
            "rc": 2.5,
        },
    )
    engine = GenericASEEngine(pot_data)
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=(12.0, 12.0, 12.0), pbc=True)
    result = engine.calculate(atoms)

    np.testing.assert_allclose(result["energies"], np.full(2, 0.5 * result["energy"]))


@pytest.mark.parametrize("engine_cls", [NumpySWEngine, NumbaSWEngine])
def test_sw_engines_skip_stress_for_nonperiodic_zero_volume_cells(engine_cls) -> None:
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
    result = engine_cls(SWData(species=["Si"])).calculate(atoms)

    assert "stress" not in result
    assert np.isfinite(result["energy"])
    assert "energies" not in result


@pytest.mark.parametrize("engine_cls", [NumpySWEngine, NumbaSWEngine])
def test_sw_engines_reject_site_energy_jacobians(engine_cls) -> None:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    engine = engine_cls(SWData(species=["Si"]))

    engine.calculate(atoms)
    with pytest.raises(NotImplementedError, match="site-energy"):
        engine.jac_energies(atoms)


@pytest.mark.parametrize("engine_cls", [NumpyADPEngine, NumbaADPEngine])
def test_adp_engines_skip_stress_for_nonperiodic_zero_volume_cells(engine_cls) -> None:
    r = np.linspace(0.1, 3.0, 5)
    rho = np.linspace(0.0, 2.0, 5)
    data = ADPData(
        species_count=1,
        r_grid=r,
        rho_grid=rho,
        phi_values=np.zeros((1, 1, len(r))),
        rho_values=np.zeros((1, 1, len(r))),
        emb_values=np.zeros((1, len(rho))),
        dipole_values=np.zeros((1, 1, len(r))),
        quadrupole_values=np.zeros((1, 1, len(r))),
    )
    data.species = np.array([13], dtype=np.int32)
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    result = engine_cls(data).calculate(atoms)

    assert "stress" not in result
    assert np.isfinite(result["energy"])


def test_write_force_handles_empty_atoms() -> None:
    from io import StringIO

    buf = StringIO()
    write_force(buf, Atoms())

    assert buf.getvalue().startswith("#N 0 1")


def test_manual_species_energy_offsets_are_added_to_predictions() -> None:
    pot_data = SWData(species=["Si"])
    pot_data.species_energy_offsets = {"Si": -1.5}

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.calc = make_calculator(pot_data, engine="numba")

    assert atoms.get_potential_energy() == pytest.approx(-1.5)
    assert "energies" not in atoms.calc.results


def test_regression_species_energy_offsets_are_fitted_from_totals() -> None:
    images = []
    for symbols, energy in [("Fe2", -8.0), ("FeC", -6.0), ("C2", -4.0)]:
        atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
        atoms.calc = SimpleNamespace(targets={"energy": energy})
        images.append(atoms)

    pot_data = SimpleNamespace(species=[26, 6], species_energy_offsets={})
    setting = LossSetting(species_energy_offset_mode="regression")

    offsets = _resolve_species_energy_offsets(images, pot_data, setting)

    assert offsets["Fe"] == pytest.approx(-4.0)
    assert offsets["C"] == pytest.approx(-2.0)


def test_species_energy_offsets_roundtrip_through_npy(tmp_path) -> None:
    pot_data = SWData(species=["Si"])
    pot_data.species_energy_offsets = {"Si": -1.5}
    path = tmp_path / "offsets.npy"

    write_potential(str(path), pot_data)
    loaded = read_potential(str(path))

    assert isinstance(loaded, SWData)
    assert loaded.species_energy_offsets == {"Si": -1.5}


def test_manual_block_freezing_works_in_python_mode_for_eam() -> None:
    data = EAMData(
        form="alloy",
        species_count=2,
        r_grid=np.array([0.1, 0.2]),
        rho_grid=np.array([0.0, 1.0]),
        phi_values=np.zeros((2, 2, 2)),
        rho_values=np.zeros((2, 2, 2)),
        emb_values=np.zeros((2, 2)),
        optimized=["pair.AlCu", "density.Cu", "embedding.Cu"],
    )
    data.species = np.array([13, 29], dtype=np.int32)

    params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    data.parameters = params

    np.testing.assert_allclose(data.phi_values[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(data.phi_values[0, 1], [1.0, 2.0])
    np.testing.assert_allclose(data.rho_values[:, 1], [[3.0, 4.0], [3.0, 4.0]])
    np.testing.assert_allclose(data.emb_values[1], [5.0, 6.0])
    assert data.number_of_parameters_optimized == 6


def test_manual_block_freezing_works_in_python_mode_for_adp() -> None:
    data = ADPData(
        form="alloy",
        species_count=2,
        r_grid=np.array([0.1, 0.2]),
        rho_grid=np.array([0.0, 1.0]),
        phi_values=np.zeros((2, 2, 2)),
        rho_values=np.zeros((2, 2, 2)),
        emb_values=np.zeros((2, 2)),
        dipole_values=np.zeros((2, 2, 2)),
        quadrupole_values=np.zeros((2, 2, 2)),
        optimized=["pair.AlCu", "dipole.AlCu", "quadrupole.CuCu"],
    )
    data.species = np.array([13, 29], dtype=np.int32)

    params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    data.parameters = params

    np.testing.assert_allclose(data.phi_values[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(data.phi_values[0, 1], [1.0, 2.0])
    np.testing.assert_allclose(data.dipole_values[0, 1], [3.0, 4.0])
    np.testing.assert_allclose(data.quadrupole_values[1, 1], [5.0, 6.0])
    assert data.number_of_parameters_optimized == 6
