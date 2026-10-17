"""Quick HPRC benchmark for a PINN-like workload: a tanh MLP (width 96, depth 4, the size of the Li-SPAN PINN) with
first and second derivatives by autograd, one Adam step per call, at batches of 4k, 32k and 256k collocation points.
Run on CPU always and on the GPU when one is visible; prints seconds per training step and the GPU/CPU ratio.

    python scripts/hpc_bench.py

Used to decide CPU job arrays vs GPU jobs for the parametric PINN (Phase 3). Not part of any result.
"""
import os
import platform
import time

import torch


def bench(device, n, width=96, depth=4, n_in=4, n_out=6, steps=20):
    torch.manual_seed(0)
    layers, d = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(width, n_out))
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    x = torch.rand(n, n_in, device=device, requires_grad=True)

    def step():
        opt.zero_grad()
        u = net(x)
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        g2 = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0]
        loss = (u ** 2).mean() + (g ** 2).mean() + (g2 ** 2).mean()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps


def main():
    print(f"python {platform.python_version()} | torch {torch.__version__} | threads {torch.get_num_threads()} "
          f"| OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    cuda = torch.cuda.is_available()
    print(f"cuda available: {cuda}" + (f" | {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}" if cuda else ""))
    for n in (4096, 32768, 262144):
        t_cpu = bench(torch.device("cpu"), n, steps=5 if n > 100000 else 20)
        line = f"batch {n:7d}: CPU {1e3 * t_cpu:9.1f} ms/step"
        if cuda:
            t_gpu = bench(torch.device("cuda"), n)
            line += f" | GPU {1e3 * t_gpu:8.1f} ms/step | GPU faster by x{t_cpu / t_gpu:.1f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
