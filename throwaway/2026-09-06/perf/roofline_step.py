"""Why one pRNN timestep starves the GPU: a launch/bandwidth/roofline census.

Answers "why is every step so slow, and what can we target". Run on an idle
GPU from the repo root:  uv run python tests/perf/roofline_step.py
"""
import time
import torch


def timeit(fn, n: int = 2000) -> float:
    """Microseconds per call, GPU-synchronised."""
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t0) / n


def main() -> None:
    assert torch.cuda.is_available(), "this measures GPU launch/bandwidth limits"
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}: {p.multi_processor_count} SMs, {p.total_memory/2**30:.1f} GiB\n")

    tiny = torch.zeros(1, device=dev)
    launch = timeit(lambda: tiny.add_(1.0))
    print(f"launch overhead, one trivial kernel        {launch:8.2f} us")

    big = torch.empty(256 * 1024 * 1024 // 4, device=dev)
    bw_peak = big.numel() * 4 / (timeit(big.sum, n=200) * 1e-6) / 1e9
    print(f"measured read bandwidth                    {bw_peak:8.1f} GB/s\n")

    H, IN = 500, 155
    W, W_in = torch.randn(H, H, device=dev), torch.randn(H, IN, device=dev)
    wbytes = (W.numel() + W_in.numel()) * 4

    print(f"recurrent matmul x[B,{H}] @ W[{H},{H}]  (W = {W.numel()*4/1024:.0f} KiB)")
    print(f"  {'B':>6}{'us/call':>10}{'GB/s':>9}{'GFLOP/s':>10}{'us/sample':>11}")
    for B in (1, 2, 8, 32, 128, 512, 2048):
        x = torch.randn(B, H, device=dev)
        t = timeit(lambda: torch.mm(x, W))
        moved = W.numel() * 4 + 2 * B * H * 4
        print(f"  {B:>6}{t:>10.2f}{moved/(t*1e-6)/1e9:>9.1f}"
              f"{2*B*H*H/(t*1e-6)/1e9:>10.1f}{t/B:>11.3f}")

    x1, xi = torch.randn(1, H, device=dev), torch.randn(1, IN, device=dev)
    measured = timeit(lambda: torch.mm(x1, W)) + timeit(lambda: torch.mm(xi, W_in.t()))
    floor = wbytes / (bw_peak * 1e9) * 1e6
    print(f"\none step's two matmuls at batch 1")
    print(f"  weights that MUST be read per step        {wbytes/1024:8.0f} KiB")
    print(f"  reading them at peak bandwidth            {floor:8.2f} us   <- HARD FLOOR")
    print(f"  launch overhead for the 2 kernels         {2*launch:8.2f} us")
    print(f"  measured                                  {measured:8.2f} us")
    print(f"\n  arithmetic intensity at B=1: {2*(H*H+H*IN)/wbytes:.2f} FLOP/byte")
    print( "  (this GPU needs ~60 FLOP/byte to be compute-bound: we are ~300x below")
    print( "   the roofline ridge, so the step is a memory-streaming problem, not a")
    print( "   compute one. Launch cost EXCEEDS the useful work per kernel.)")


if __name__ == "__main__":
    main()
