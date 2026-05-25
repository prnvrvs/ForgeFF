"""IO unils."""

import logging
from collections.abc import Iterable
from numbers import Integral

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

import forgeff.io
from forgeff.parallel import DummyMPIComm, is_master, world

logger = logging.getLogger(__name__)


def get_dummy_species(images: list[Atoms]) -> list[int]:
    """Get dummy species particularly for images read from `.cfg` files.

    Returns
    -------
    list[int]

    """
    m = 0
    for atoms in images:
        m = max(m, atoms.numbers.max())
    return list(range(m + 1))


def _species_labels(species: Iterable[int | str] | None) -> list[str]:
    if species is None:
        return []
    labels: list[str] = []
    for item in species:
        if isinstance(item, Integral):
            labels.append(str(chemical_symbols[int(item)]))
        else:
            labels.append(str(item))
    return labels


def set_potential_species(pot_data: object, species: Iterable[int | str] | None) -> None:
    """Assign species to a potential object using the representation it accepts."""
    labels = _species_labels(species)
    try:
        setattr(pot_data, "species", labels)
        return
    except (TypeError, ValueError):
        pass

    numbers = [int(atomic_numbers[label]) for label in labels]
    setattr(pot_data, "species", numbers)


def read_images(
    filenames: list[str],
    species: list[int] | None = None,
    comm: DummyMPIComm = world,
    title: str = "configurations",
) -> list[Atoms]:
    """Read images.

    Returns
    -------
    list[Atoms]

    """
    images = []
    if is_master(comm):
        logger.info("%s\n", "=" * 72)
        logger.info("[%s]", title)
        for filename in filenames:
            images_local = forgeff.io.read(filename, species)
            images.extend(images_local)
            logger.info('"%s" = %s', filename, len(images_local))
        logger.info("")
    return comm.bcast(images, root=0)
