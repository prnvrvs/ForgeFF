import numpy as np
import pytest
from ase import Atoms
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.calculator import make_calculator
from forgeff.parallel import world

def verify_forces(atoms, eps=1e-6, rtol=1e-4, atol=1e-5):
    """Generic helper to verify analytical forces against numerical ones."""
    f_analytical = atoms.get_forces()
    f_numerical = np.zeros_like(f_analytical)
    
    e0 = atoms.get_potential_energy()
    for i in range(len(atoms)):
        for j in range(3):
            pos = atoms.positions[i, j]
            atoms.positions[i, j] += eps
            e_plus = atoms.get_potential_energy()
            atoms.positions[i, j] = pos # reset
            f_numerical[i, j] = -(e_plus - e0) / eps
            
    np.testing.assert_allclose(f_analytical, f_numerical, rtol=rtol, atol=atol)

def test_adp_forces_pure():
    """Test ADP analytical forces for a pure element."""
    nr = 50
    nrho = 50
    pot_data = ADPData()
    pot_data.species = [13]
    pot_data.r_grid = np.linspace(0.1, 6.0, nr)
    pot_data.rho_grid = np.linspace(0.0, 15.0, nrho)
    
    rng = np.random.default_rng(42)
    pot_data.initialize(rng)
    
    atoms = Atoms('Al3', positions=[
        [0.0, 0.0, 0.0],
        [2.5, 0.5, 0.1],
        [1.0, 2.0, -0.5]
    ], cell=[10, 10, 10], pbc=True)
    
    atoms.calc = make_calculator(pot_data, engine='numba')
    verify_forces(atoms)

def test_adp_forces_alloy():
    """Test ADP analytical forces for a binary alloy."""
    nr = 50
    nrho = 50
    pot_data = ADPData()
    pot_data.species = [13, 14] # Al, Si
    pot_data.r_grid = np.linspace(0.1, 6.0, nr)
    pot_data.rho_grid = np.linspace(0.0, 15.0, nrho)
    
    rng = np.random.default_rng(123)
    pot_data.initialize(rng)
    
    atoms = Atoms('AlSi', positions=[
        [0.0, 0.0, 0.0],
        [2.2, 0.8, 0.3]
    ], cell=[10, 10, 10], pbc=True)
    
    atoms.calc = make_calculator(pot_data, engine='numba')
    verify_forces(atoms)


def test_adp_write_roundtrip_uses_dataclass_state(tmp_path):
    pot_data = ADPData()
    pot_data.species = [13]
    pot_data.r_grid = np.linspace(0.1, 6.0, 10)
    pot_data.rho_grid = np.linspace(0.0, 15.0, 10)
    pot_data.initialize(np.random.default_rng(7))

    filename = tmp_path / "adp.npy"
    pot_data.write(filename)

    loaded = np.load(filename, allow_pickle=True).item()
    assert "optimized" in loaded
    assert "_species" in loaded
    assert "calculate" not in loaded


def test_adp_fs_runtime_is_rejected_explicitly():
    pot_data = ADPData(form="fs")
    pot_data.species = [13, 29]
    pot_data.r_grid = np.linspace(0.1, 6.0, 10)
    pot_data.rho_grid = np.linspace(0.0, 15.0, 10)
    pot_data.initialize(np.random.default_rng(7))

    with pytest.raises(NotImplementedError, match="Finnis-Sinclair"):
        make_calculator(pot_data, engine='numba')

if __name__ == "__main__":
    test_adp_forces_pure()
    test_adp_forces_alloy()
    if world.rank == 0:
        print("All ADP Force tests passed!")
