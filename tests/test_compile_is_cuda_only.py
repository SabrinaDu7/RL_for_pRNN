"""`train_prnn.compile` must not trace when the module is not on CUDA.

Every eval in this repo runs under `on_device(..., "cpu")`, and a compiled
callable follows its module: device is a dynamo guard, so a compiled recurrence
called on CPU retraces its whole unrolled loop. Measured 2026-08-28 on
`multienv-impassable-traj`: one analysis event emitted 76,205 dynamo warning
lines and the event costs ~88 s against ~37 s uncompiled.

The compile was CUDA-only at construction already
(`prnn_adapter.py`, `if compile_cell and self.device.type == "cuda"`); these
tests pin that it stays CUDA-only at CALL time too, which is the part
`on_device` can undo.
"""

import pytest
import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from curious_george.models.device import on_device
from curious_george.models.prnn_adapter import _compile_on_cuda_only


class Recurrence(torch.nn.Module):
    """Stands in for the pRNN layer: a python loop dynamo unrolls."""

    def __init__(self, width: int = 4, steps: int = 3):
        super().__init__()
        self.cell = torch.nn.Linear(width, width)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            x = self.cell(x).relu()
        return x


@pytest.fixture
def clean_dynamo():
    dynamo.reset()
    counters.clear()
    yield
    dynamo.reset()
    counters.clear()


def graphs_compiled() -> int:
    return counters["stats"]["unique_graphs"]


def test_cpu_module_never_traces(clean_dynamo):
    """The whole point: no dynamo graph is built for a CPU forward."""
    m = Recurrence()
    _compile_on_cuda_only(m, dynamic=False)

    m(torch.randn(2, 4))

    assert graphs_compiled() == 0


def test_eager_and_compiled_agree_bitwise(clean_dynamo):
    """Dispatching to eager must not change the value, only the path."""
    m = Recurrence()
    x = torch.randn(2, 4)
    expected = m(x)

    _compile_on_cuda_only(m, dynamic=False)

    assert torch.equal(m(x), expected)


def test_forward_keeps_its_identity(clean_dynamo):
    """`functools.wraps`: the wrapper must not erase what it wraps."""
    m = Recurrence()
    _compile_on_cuda_only(m, dynamic=False)

    assert m.forward.__name__ == "forward"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_module_does_trace(clean_dynamo):
    """The negative control: on CUDA the compile must still happen, or this
    'fix' would be a silent removal of a 3.89x speedup."""
    m = Recurrence().cuda()
    _compile_on_cuda_only(m, dynamic=False)

    m(torch.randn(2, 4, device="cuda"))

    assert graphs_compiled() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_device_cpu_round_trip_traces_once(clean_dynamo):
    """The measured failure, end to end.

    A CUDA module compiles on its first call. `on_device(..., "cpu")` - what
    every eval does - must then add NO further graphs, and the module must go
    back to the compiled path afterwards.
    """
    m = Recurrence().cuda()
    _compile_on_cuda_only(m, dynamic=False)

    m(torch.randn(2, 4, device="cuda"))
    after_warmup = graphs_compiled()
    assert after_warmup > 0

    with on_device(m, "cpu"):
        m(torch.randn(2, 4))
    assert graphs_compiled() == after_warmup, "the CPU eval retraced"

    m(torch.randn(2, 4, device="cuda"))
    assert graphs_compiled() == after_warmup, "the return to CUDA retraced"


def test_the_unfixed_pattern_does_trace_on_cpu(clean_dynamo):
    """Why the helper exists, pinned so it cannot quietly stop being true.

    This is the code that shipped before 2026-08-28: `torch.compile` bound to
    `forward` with no device condition. It traces on CPU, which is the whole
    defect - `test_cpu_module_never_traces` above is the same call through the
    helper and must stay at zero.
    """
    m = Recurrence()
    m.forward = torch.compile(m.forward, dynamic=False)

    m(torch.randn(2, 4))

    assert graphs_compiled() > 0
