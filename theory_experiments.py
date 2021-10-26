import argparse
import warnings

import lab.torch as B
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.out as out
from scipy.special import loggamma
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop

from pacbayes.convex import CatoniMixture, BernoulliKL, Convex
from pacbayes.utils import device

out.key_width = 15
B.device(str(device)).__enter__()

delta = torch.tensor(0.1).to(device)
n = 30


def phi(x):
    return 1 - 0.5 * B.sqrt(1 - x ** 2)


def best_catoni_parameters():
    best_cs = []
    for alpha, q in zip((kl_dist - B.log(delta)) / n, q_dist):
        cs = B.linspace(torch.float64, 0, 20, 10_000)
        vals = (1 - B.exp(-cs * q - alpha)) / (1 - B.exp(-cs))
        best_cs.append(cs[torch.argmin(vals)])
    return B.stack(*best_cs)


def best_expected_catoni_parameter():
    alpha = (kl_dist - B.log(delta)) / n
    cs = B.linspace(torch.float64, 0, 20, 10_000)[:, None]
    vals = B.mean(
        (1 - B.exp(-cs * q_dist[None, :] - alpha[None, :])) / (1 - B.exp(-cs)),
        axis=1,
    )
    return cs[torch.argmin(vals), 0]


def compute_log_r(convex_delta, r=None):
    # Estimate the supremum by taking the maximum over a dense `linspace`.
    if r is None:
        r = torch.linspace(1e-6, 1 - 1e-6, 10_000).to(device)
    else:
        if B.isscalar(r):
            r = r[None]
    k_over_ms = torch.linspace(0, 1, n + 1).to(device)  # [n + 1]

    # Precompute the values for delta in the terms of the sum. We want to keep
    # the computation graph lean.
    convex_risks = B.reshape(
        convex_delta(
            B.stack(
                B.reshape(B.tile(k_over_ms[:, None], 1, len(r)), -1),
                B.reshape(B.tile(r[None, :], n + 1, 1), -1),
                axis=1,
            )
        ),
        n + 1,
        len(r),
    )  # [n + 1, R]

    log_terms = []
    for k in range(0, n + 1):
        logcomb = loggamma(n + 1) - loggamma(n - k + 1) - loggamma(k + 1)
        log_pmf = logcomb + k * B.log(r) + (n - k) * B.log(1 - r)
        delta = convex_risks[k, :]
        log_terms.append(log_pmf + n * delta)
    log_supremum_r = torch.logsumexp(B.stack(*log_terms, axis=1), dim=1)

    return r, log_supremum_r


def _convert(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).to(device)
    if B.isscalar(x):
        x = x[None]
    return x


def compute_bound(convex_delta, q, kl, illegal=False):
    q = _convert(q)
    kl = _convert(kl)
    _, log_r = compute_log_r(convex_delta)
    if illegal:
        alpha = (kl - B.log(delta)) / n
    else:
        alpha = (kl + torch.max(log_r) - B.log(delta)) / n
    values = convex_delta.biggest_inverse(q, alpha)
    check = B.min(alpha - convex_delta(B.stack(q, values, axis=1)))
    if check < -1e-6:
        warnings.warn(f"Check violated! Discrepancy: {check}!")
    return values


def compute_expected_bound(convex_delta, illegal=False):
    return B.mean(compute_bound(convex_delta, q_dist, kl_dist, illegal=illegal))


parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true")
parser.add_argument("--reps", type=int, default=10)
parser.add_argument("--plot-deltas", action="store_true")
parser.add_argument(
    "--setting",
    choices=[
        "det1-1",
        "det1-2",
        "det2-1",
        "det2-2",
        "stoch1",
        "stoch2",
        "stoch3",
        "random",
    ],
    required=True,
)
parser.add_argument("--random-seed", type=int)
parser.add_argument("--random-better-bound", choices=["maurer", "catoni"])
args = parser.parse_args()


if args.setting.startswith("stoch"):
    wd = WorkingDirectory("_experiments", "theory", args.setting, seed=0)

    if args.setting == "stoch1":
        args.rate = 5e-3
        args.iters = 10_000
        args.units = 512
    elif args.setting in {"stoch2", "stoch3"}:
        args.rate = 5e-3
        args.iters = 100_000
        args.units = 1024
    else:
        raise AssertionError(f"Undefined setting {args.setting}.")

    q_dist = torch.tensor(
        {
            "stoch1": [0.02, 0.05],
            "stoch2": [0.3, 0.4],
            "stoch3": [0.35, 0.45, 0.40, 0.43],
        }[args.setting]
    ).to(device)
    kl_dist = torch.tensor(
        {
            "stoch1": [1, 2],
            "stoch2": [1, 50],
            "stoch3": [5, 30, 7, 25],
        }[args.setting]
    ).to(device)
elif args.setting.startswith("det"):
    wd = WorkingDirectory("_experiments", "theory", args.setting, seed=0)

    if args.setting.startswith("det1"):
        args.rate = 5e-3
        args.iters = 500
        args.units = 256
    elif args.setting.startswith("det2"):
        args.rate = 5e-3
        args.iters = 1500
        args.units = 256
    else:
        raise AssertionError(f"Undefined setting {args.setting}.")

    q_dist = torch.tensor(
        {
            "det1-1": [0.02],
            "det1-2": [0.05],
            "det2-1": [0.3],
            "det2-2": [0.4],
        }[args.setting]
    ).to(device)
    kl_dist = torch.tensor(
        {
            "det1-1": [1],
            "det1-2": [2],
            "det2-1": [1],
            "det2-2": [50],
        }[args.setting]
    ).to(device)
elif args.setting == "random":
    if args.random_seed is None:
        raise ValueError("Must set --random-seed.")
    if args.random_better_bound is None:
        raise ValueError("Must set --random-better-bound.")
    wd = WorkingDirectory(
        "_experiments", "theory", "random", str(args.random_seed), seed=args.random_seed
    )
    args.setting = f"random-{args.random_seed}"

    args.reps = 1
    args.rate = 5e-3
    args.iters = 1_000_000
    args.units = 1024

    out.out("Rejection sampling...")
    atoms = 3

    while True:
        q_dist = 0.02 + (0.2 - 0.02) * B.rand(np.float32, atoms)
        kl_dist = 0.5 + (50 - 0.5) * B.rand(np.float32, atoms)

        q_dist = torch.tensor(q_dist).to(device)
        kl_dist = torch.tensor(kl_dist).to(device)
        alpha = (kl_dist - B.log(delta)) / n

        maurer = BernoulliKL()
        catoni = CatoniMixture(best_expected_catoni_parameter())
        maurer_bound = B.to_numpy(compute_expected_bound(maurer))
        catoni_bound = B.to_numpy(compute_expected_bound(catoni))

        # Perform rejection sampling.

        if B.abs(maurer_bound - catoni_bound) < 0.002:
            continue

        if maurer_bound < catoni_bound and args.random_better_bound == "catoni":
            continue

        if catoni_bound < maurer_bound and args.random_better_bound == "maurer":
            continue

        if max(maurer_bound, catoni_bound) > 0.8:
            continue

        # Success!
        break

    with out.Section("Chosen sample"):
        out.kv("Dist. for q", B.to_numpy(q_dist))
        out.kv("Dist. for KL", B.to_numpy(kl_dist))
        out.kv("Maurer - Catoni", maurer_bound - catoni_bound)


else:
    raise AssertionError(f"Undefined setting {args.setting}.")


alpha = (kl_dist - B.log(delta)) / n

out.kv("Best expected Catoni parameter", B.to_numpy(best_expected_catoni_parameter()))
maurer = BernoulliKL()
catoni = CatoniMixture(best_expected_catoni_parameter())

with out.Section("Bounds"):
    out.kv("Catoni", B.to_numpy(compute_expected_bound(catoni)))
    out.kv("Maurer", B.to_numpy(compute_expected_bound(maurer)))
    out.kv("Illegal Maurer", B.to_numpy(compute_expected_bound(maurer, illegal=True)))


def optimise_convex():
    convex = Convex(init_iters=0, n_hidden=args.units).to(device)
    opt = torch.optim.Adam(params=convex.parameters(), lr=args.rate)
    iters = args.iters
    bounds = []
    with out.Progress("Optimising convex function", total=iters) as progress:
        try:
            for _ in range(iters):
                bound = compute_expected_bound(convex)
                bound.backward()
                opt.step()
                opt.zero_grad()
                bounds.append(B.to_numpy(bound))
                progress(bound=B.to_numpy(bound))
        except KeyboardInterrupt:
            pass

    return convex, np.array(bounds)


if args.load:
    record = wd.load(f"record.pickle")
    deltas = []
    for i in range(args.reps):
        convex = Convex(init_iters=0, n_hidden=args.units)
        convex.load_state_dict(torch.load(wd.file(f"convex{i}.pt"), map_location="cpu"))
        deltas.append(convex)

else:
    record = []
    deltas = []
    with out.Progress("Performing repetitions", total=args.reps) as progress:
        for i in range(args.reps):
            progress()
            convex, bounds = optimise_convex()
            torch.save(convex.state_dict(), wd.file(f"convex{i}.pt"))
            deltas.append(convex.to("cpu"))
            record.append(bounds)
    record = B.to_numpy(B.stack(*record, axis=0))
    wd.save(record, f"record.pickle")

# Compute all bounds.
maur_bound = B.to_numpy(compute_expected_bound(maurer))
cat_bound = B.to_numpy(compute_expected_bound(catoni))
ill_bound = B.to_numpy(compute_expected_bound(maurer, illegal=True))

# Determine reference.
if args.setting.startswith("det"):
    ref = cat_bound
else:
    ref = ill_bound

# Compute values for plot.
x = np.arange(1, B.shape(record)[1] + 1)
record = record - ref
mean = np.mean(record, axis=0)
lower = np.min(record, axis=0)
upper = np.max(record, axis=0)

# Configure latex for legends.
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "sans-serif"})
params = {"text.latex.preamble": [r"\usepackage{amsmath}", r"\usepackage{amsfonts}"]}
plt.rcParams.update(params)


plt.figure(figsize=(3, 2.5))
plt.plot(x, mean, label="Learned convex")
plt.fill_between(x, lower, upper, alpha=0.3, facecolor="tab:blue")
cat_par = B.to_numpy(best_expected_catoni_parameter())
if not args.setting.startswith("det"):
    plt.axhline(
        cat_bound - ref,
        label=f"Optimal Catoni ($\\beta = {cat_par:.2f}$)",
        c="tab:orange",
        ls="--",
    )
plt.axhline(
    maur_bound - ref,
    label="Maurer",
    c="tab:red",
    ls=":",
)
plt.xlabel("Iteration")
if args.setting.startswith("det"):
    plt.ylabel("$\\overline{p}_{\\Delta} - \\inf_{\\beta>0}\\overline{p}_{C_\\beta}$")
else:
    plt.ylabel("$\\mathbb{E}[\\overline{p}_{\\Delta}] - \\mathbb{E}[\\underline{p}]$")
plt.gca().get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda x, p: str(int(x)) if x < 1000 else f"{int(x / 1000)}k"
    )
)
plt.semilogx()
plt.semilogy()
tweak(legend=False)
plt.savefig(wd.file(f"{args.setting}_graph.pdf"))
pdfcrop(wd.file(f"{args.setting}_graph.pdf"))
plt.close()

if args.plot_deltas:
    for i in range(args.reps):
        plt.figure(figsize=(6, 2.5))
        X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z_convex = B.to_numpy(
            deltas[i](
                torch.tensor(
                    B.stack(X.reshape(-1), Y.reshape(-1), axis=1), dtype=torch.float32
                )
            )
        ).reshape(100, 100)
        Z_convex -= Z_convex.mean()
        Z_cat = B.to_numpy(
            catoni(
                torch.tensor(
                    B.stack(X.reshape(-1), Y.reshape(-1), axis=1), dtype=torch.float32
                )
            )
        ).reshape(100, 100)
        Z_cat -= Z_cat.mean()

        levels = B.linspace(
            min(Z_convex.min(), Z_cat.min()), max(Z_convex.max(), Z_cat.max()), 20
        )

        plt.subplot(1, 2, 1)
        plt.title("Catoni")
        plt.contourf(X, Y, Z_cat, levels)
        plt.xlabel("$q$")
        plt.ylabel("$p$")
        tweak(grid=False)

        plt.subplot(1, 2, 2)
        plt.title("Learned Convex")
        plt.contourf(X, Y, Z_convex, levels)
        plt.xlabel("$q$")
        plt.ylabel("$p$")
        tweak(grid=False)

        plt.savefig(wd.file(f"{args.setting}_convex{i}.pdf"))
        pdfcrop(wd.file(f"{args.setting}_convex{i}.pdf"))
        plt.close()
