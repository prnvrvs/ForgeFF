Engines
=======

ForgeFF keeps the runtime engine in the training setting file under
``[common].engine``.

Use ``make_calculator(potential_data, engine=...)`` in Python, or set
``[common].engine`` in the matching ``forgeff.train.toml`` file.

Public engine names
-------------------

- ``ASE``: adapter path for supported ASE calculators and direct ASE classes
- ``numpy``: **ForgeFF NumPy**
- ``numba``: **ForgeFF Numba**

ASE calculators are external calculators that ForgeFF can call through the
adapter path. ForgeFF's own engines are **ForgeFF NumPy** and **ForgeFF Numba**.

Supported combinations
----------------------

- Analytical pair potentials:
  - ``ASE`` for built-in ASE-compatible forms
  - ``numpy`` for the ForgeFF reference path
  - ``numba`` for the ForgeFF JIT path
- EAM:
  - ``ASE``
  - ``numpy``
  - ``numba``
- ADP:
  - ``numpy``
  - ``numba``
- Stillinger-Weber:
  - ``numpy``
  - ``numba``
- Tersoff:
  - ``numba``

If you need a LAMMPS output file, see :doc:`cli/export` for tabulated EAM
and ADP potentials. Analytical pair potentials and Stillinger-Weber are
runtime engines only. Tersoff can be loaded from TOML or the Python API and
still executes through the Numba engine.

Usage examples
--------------

Pairwise analytical Morse
^^^^^^^^^^^^^^^^^^^^^^^^^

TOML first:

.. code-block:: toml

    [potential]
    family = "analytical"
    form = "morse"
    cutoff = 8.0

    [species]
    order = ["Al"]

    [pair.AlAl]
    # Morse is the simplest built-in pair form in the registry.
    # Parameter order: De, a, re
    initial = [0.20, 1.50, 2.75]

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numpy">ForgeFF NumPy</span>

.. code-block:: python

    from ase.build import bulk
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("examples/toml/pairwise/morse/unary/initial.toml")
    atoms = bulk("Al", "fcc", a=4.05)
    atoms.calc = make_calculator(pot, engine="numpy")

.. code-block:: text

    energy = -3.114291273940021
    force norm = 0.0
    stress norm = 0.28056336957738404

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numba">ForgeFF Numba</span>

.. code-block:: python

    from ase.build import bulk
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("examples/toml/pairwise/morse/unary/initial.toml")
    atoms = bulk("Al", "fcc", a=4.05)
    atoms.calc = make_calculator(pot, engine="numba")

.. code-block:: text

    energy = -3.114291273940021
    force norm = 0.0
    stress norm = 0.2805633695773841

EAM alloy
^^^^^^^^^

Tabulated EAM fits can freeze individual pair, density, or embedding blocks by
setting ``optimize = false`` in TOML, or by selecting the block names in
``pot.optimized`` when using the Python API.

.. raw:: html

   <span class="ff-engine-label ff-engine-label--matscipy">ASE</span>

.. code-block:: python

    from ase.build import bulk
    from ase.calculators.eam import EAM
    import numpy as np

    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = EAM(potential="tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy")

.. code-block:: text

    energy = -221.5222074023863
    force norm = 2.2131806604481135
    stress norm = 0.017547255220923076

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numpy">ForgeFF NumPy</span>

.. code-block:: python

    from ase.build import bulk
    import numpy as np
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy", form="alloy")
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = make_calculator(pot, engine="numpy")

.. code-block:: text

    energy = -221.52220740238633
    force norm = 2.213180660448107
    stress norm = 0.017547255220916665

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numba">ForgeFF Numba</span>

.. code-block:: python

    from ase.build import bulk
    import numpy as np
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy", form="alloy")
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = make_calculator(pot, engine="numba")

.. code-block:: text

    energy = -221.52220740238633
    force norm = 2.213180660448107
    stress norm = 0.017547255220916665

ADP
^^^

Tabulated ADP fits use the same freeze convention as EAM, with the additional
``dipole`` and ``quadrupole`` blocks available to freeze or optimize
independently.

.. raw:: html

   <span class="ff-engine-label ff-engine-label--matscipy">ASE</span>

.. code-block:: python

    from ase.build import bulk
    from ase.calculators.eam import EAM
    import numpy as np

    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    rng = np.random.default_rng(11)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = EAM(potential="tests/data_path/nist/AlCu.adp", form="adp")

.. code-block:: text

    energy = -26.876415872339763
    force norm = 0.23637488863847092
    stress norm = 9.852717852070938e-05

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numpy">ForgeFF NumPy</span>

.. code-block:: python

    from ase.build import bulk
    import numpy as np
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("tests/data_path/nist/AlCu.adp", form="adp")
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    rng = np.random.default_rng(11)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = make_calculator(pot, engine="numpy")

.. code-block:: text

    energy = -26.876415872339756
    force norm = 0.23637488863846734
    stress norm = 9.852717852049338e-05

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numba">ForgeFF Numba</span>

.. code-block:: python

    from ase.build import bulk
    import numpy as np
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("tests/data_path/nist/AlCu.adp", form="adp")
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    rng = np.random.default_rng(11)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.calc = make_calculator(pot, engine="numba")

.. code-block:: text

    energy = -26.876415872339756
    force norm = 0.23637488863846728
    stress norm = 9.85271785204985e-05

Stillinger-Weber
^^^^^^^^^^^^^^^^

.. raw:: html

   <span class="ff-engine-label ff-engine-label--matscipy">Matscipy</span>

.. code-block:: python

    from ase.build import bulk
    from matscipy.calculators.manybody import StillingerWeber
    from matscipy.calculators.manybody.calculator import Manybody

    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    params = {
        "el": "Si",
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "costheta0": 1.0 / 3.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "a": 1.8,
        "lambda1": 21.0,
        "gamma": 1.2,
    }
    sw = StillingerWeber(params)
    calc = Manybody(
        sw["atom_type"],
        sw["pair_type"],
        sw["F"],
        sw["G"],
        sw["d1F"],
        sw["d2F"],
        sw["d11F"],
        sw["d22F"],
        sw["d12F"],
        sw["d1G"],
        sw["d11G"],
        sw["d2G"],
        sw["d22G"],
        sw["d12G"],
        sw["cutoff"],
    )
    atoms.calc = calc

.. code-block:: text

    energy = -69.38557207743135
    force norm = 0.0
    stress norm = 0.0005757386107418878

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numpy">ForgeFF NumPy</span>

.. code-block:: python

    from ase.build import bulk
    from forgeff.calculator import make_calculator
    from forgeff.potentials.sw.data import SWData

    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    pot = SWData(
        species=["Si"],
        epsilon=2.1683,
        sigma=2.0951,
        costheta0=1.0 / 3.0,
        A=7.049556277,
        B=0.6022245584,
        p=4.0,
        a=1.8,
        lambda1=21.0,
        gamma=1.2,
    )
    atoms.calc = make_calculator(pot, engine="numpy")

.. code-block:: text

    energy = -69.38557207743135
    force norm = 0.0
    stress norm = 0.0005757386107418886

.. raw:: html

   <span class="ff-engine-label ff-engine-label--forgeff-numba">ForgeFF Numba</span>

.. code-block:: python

    from ase.build import bulk
    from forgeff.calculator import make_calculator
    from forgeff.potentials.sw.data import SWData

    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    pot = SWData(
        species=["Si"],
        epsilon=2.1683,
        sigma=2.0951,
        costheta0=1.0 / 3.0,
        A=7.049556277,
        B=0.6022245584,
        p=4.0,
        a=1.8,
        lambda1=21.0,
        gamma=1.2,
    )
    atoms.calc = make_calculator(pot, engine="numba")

.. code-block:: text

    energy = -69.38557207743135
    force norm = 0.0
    stress norm = 0.0005757386107418886

Example
-------

.. code-block:: python

    from ase.build import bulk
    from forgeff.calculator import make_calculator
    from forgeff.io import read_potential

    pot = read_potential("tests/data_path/nist/Al99.eam.alloy", form="alloy")
    atoms = bulk("Al", "fcc", a=4.05)
    atoms.calc = make_calculator(pot, engine="numba")

    print("energy =", atoms.get_potential_energy())
    print("forces shape =", atoms.get_forces().shape)
    print("stress shape =", atoms.get_stress().shape)

.. code-block:: text

    energy = -3.3599999881349643
    forces shape = (1, 3)
    stress shape = (6,)

For the file format, see :doc:`toml`. For the equations, see :doc:`theory`
and :doc:`analytical`.
