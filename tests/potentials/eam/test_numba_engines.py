from __future__ import annotations

import numpy as np
from ase.build import bulk

from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.eam.numpy.engine import ASEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.engine import NumbaEAMEngine


def _make_eam_data() -> EAMData:
    r = np.linspace(1.5, 8.0, 128)
    rho = np.linspace(0.0, 6.0, 128)
    phi = np.zeros((1, 1, r.size))
    rho_vals = np.zeros((1, 1, r.size))
    emb = np.zeros((1, rho.size))
    phi[0, 0] = 0.02 * np.exp(-0.8 * (r - r[0]))
    rho_vals[0, 0] = 0.1 * np.exp(-0.5 * (r - r[0]))
    emb[0] = -0.01 * np.sqrt(rho + 1e-12)
    data = EAMData(r_grid=r, rho_grid=rho, phi_values=phi, rho_values=rho_vals, emb_values=emb)
    data.species = np.array([13], dtype=np.int32)
    return data


def _make_adp_data(*, angular: bool) -> ADPData:
    r = np.linspace(1.5, 8.0, 128)
    rho = np.linspace(0.0, 6.0, 128)
    phi = np.zeros((1, 1, r.size))
    rho_vals = np.zeros((1, 1, r.size))
    emb = np.zeros((1, rho.size))
    dip = np.zeros((1, 1, r.size))
    quad = np.zeros((1, 1, r.size))
    phi[0, 0] = 0.02 * np.exp(-0.8 * (r - r[0]))
    rho_vals[0, 0] = 0.1 * np.exp(-0.5 * (r - r[0]))
    emb[0] = -0.01 * np.sqrt(rho + 1e-12)
    if angular:
        dip[0, 0] = 0.03 * np.exp(-0.7 * (r - r[0]))
        quad[0, 0] = 0.04 * np.exp(-0.6 * (r - r[0]))
    data = ADPData(
        r_grid=r,
        rho_grid=rho,
        phi_values=phi,
        rho_values=rho_vals,
        emb_values=emb,
        dipole_values=dip,
        quadrupole_values=quad,
    )
    data.species = np.array([13], dtype=np.int32)
    return data


def test_numba_eam_matches_ase_reference() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    data = _make_eam_data()

    ase_engine = ASEAMEngine(data)
    numba_engine = NumbaEAMEngine(data)

    ase_res = ase_engine.calculate(atoms)
    numba_res = numba_engine.calculate(atoms)

    np.testing.assert_allclose(numba_res["energy"], ase_res["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_res["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_res["stress"], rtol=1e-12, atol=1e-12)


def test_ase_eam_jacobian_matches_numba_finite_difference() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    data = _make_eam_data()

    ase_engine = ASEAMEngine(data)
    numba_engine = NumbaEAMEngine(data)

    ase_jac = ase_engine.jac_energy(atoms).parameters
    numba_jac = numba_engine.jac_energy(atoms).parameters

    np.testing.assert_allclose(ase_jac, numba_jac, rtol=1e-6, atol=1e-6)


def test_numba_adp_matches_eam_when_angular_terms_are_zero() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    eam_data = _make_eam_data()
    adp_data = _make_adp_data(angular=False)

    eam_engine = NumbaEAMEngine(eam_data)
    adp_engine = NumbaADPEngine(adp_data)

    eam_res = eam_engine.calculate(atoms)
    adp_res = adp_engine.calculate(atoms)

    np.testing.assert_allclose(adp_res["energy"], eam_res["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adp_res["forces"], eam_res["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adp_res["stress"], eam_res["stress"], rtol=1e-12, atol=1e-12)
