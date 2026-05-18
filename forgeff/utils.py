"""Utilities."""

import contextlib
import os
import pathlib
import time
import typing

import logging
import sys
from forgeff.parallel import DummyMPIComm, is_master, master, world


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging.

    Parameters
    ----------
    level : int
        Logging level.

    """
    root = logging.getLogger()
    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    if master:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        root.addHandler(handler)
        root.setLevel(level)
    else:
        root.setLevel(logging.ERROR)
        root.addHandler(logging.NullHandler())


@contextlib.contextmanager
def cd(path: str | pathlib.Path) -> typing.Generator:
    """Change directory temporalily.

    Parameters
    ----------
    path: Path
        Path to directory.

    """
    cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def measure_time(name: str, *, comm: DummyMPIComm = world) -> typing.Generator:
    """Measure time.

    Parameters
    ----------
    name : str
        Name of the block.
    comm : MPI.Comm
        MPI communicator.

    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        if is_master(comm):
            print(f"Time ({name}): {end_time - start_time} (s)\n", flush=True)
