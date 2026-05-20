from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from forgeff.optimizers.base import ParallelOptimizerBase


class _RecordingComm:
    def __init__(self) -> None:
        self.rank = 0
        self.size = 1
        self.broadcasts: list[object] = []

    def bcast(self, obj, root: int = 0):
        self.broadcasts.append(obj)
        return obj

    def allreduce(self, sendobj, op=None):
        return sendobj

    def Allreduce(self, sendobj=None, recvobj=None, op=None):
        recvobj[...] = sendobj[...]


class _DummyLoss:
    def __init__(self) -> None:
        self.comm = _RecordingComm()
        self.pot_data = SimpleNamespace(parameters=np.array([1.0]), optimized=[])

    def __call__(self, parameters):
        return 0.0


class _FailingOptimizer(ParallelOptimizerBase):
    @property
    def optimized_default(self) -> list[str]:
        return []

    @property
    def optimized_allowed(self) -> list[str]:
        return []

    def _optimize(self, **kwargs):
        raise RuntimeError("boom")


def test_parallel_optimizer_broadcasts_stop_on_exception() -> None:
    loss = _DummyLoss()
    optimizer = _FailingOptimizer(loss)

    with pytest.raises(RuntimeError, match="boom"):
        optimizer.optimize()

    assert optimizer.loss.comm.broadcasts[-1] == optimizer._OP_STOP
