from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


import os

class DummyMPIComm:
    """Dummy MPI communicator.

    https://github.com/mpi4py/mpi4py/blob/master/src/mpi4py/MPI.pyi
    """

    def __init__(self) -> None:
        self.rank = 0
        self.size = 1
        # Detect physical rank for logging purposes
        self._physical_rank = int(
            os.environ.get("OMPI_COMM_WORLD_RANK",
            os.environ.get("PMI_RANK",
            os.environ.get("MV2_COMM_WORLD_RANK",
            os.environ.get("SLURM_PROCID",
            os.environ.get("RANK", 0)))))
        )

    @property
    def master(self) -> bool:
        """Return True if this is the master process (rank 0)."""
        return self._physical_rank == 0

    def barrier(self) -> None: ...

    def send(self, obj: Any, dest: int, tag: int = 0) -> None: ...

    def recv(self, buf: Any, source: int = 0, tag: int = 0) -> Any:
        return buf

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def gather(self, sendobj: Any, root: int = 0) -> list[Any]:
        return [sendobj]

    def scatter(self, sendobj: Sequence[Any], root: int = 0) -> Any:
        return sendobj[0]

    def allgather(self, sendobj: Any) -> list[Any]:
        return [sendobj]

    def Allreduce(self, sendobj: Any = None, recvobj: Any = None, op=None) -> None:
        recvobj[...] = sendobj[...]

    def allreduce(self, sendobj: Any, op=None) -> Any:
        return sendobj


def _get_world() -> MPI.Comm | DummyMPIComm:
    """Get the world MPI communicator depending on if `mpi4py` is installed.

    Returns
    -------
    MPI.Comm | DummyMPIComm
        World MPI communicator.

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return DummyMPIComm()
    return MPI.COMM_WORLD


world = _get_world()


def is_master(comm: Any = world) -> bool:
    """Return True if the communicator is at rank 0."""
    if hasattr(comm, "master"):
        return comm.master
    return comm.rank == 0


master = is_master(world)
