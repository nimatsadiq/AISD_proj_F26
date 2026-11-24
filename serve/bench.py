"""Latency benchmark for the prediction service.

Rubric 5 asks for latency and cost benchmarks, so this produces numbers you
can read out rather than an adjective. Run it against localhost first, then
against the deployed URL, and quote both: the gap between them is network,
not compute, which is a point worth being able to make.

    python -m serve.bench                              # local
    python -m serve.bench --url http://<EC2-IP>        # deployed
    python -m serve.bench --url ... --n 200 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def time_one(url: str, timeout: float = 10.0) -> float:
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        r.read()
    return (time.perf_counter() - start) * 1000.0


def bench(base: str, path: str, n: int, warmup: int = 5) -> dict:
    url = base.rstrip("/") + path
    for _ in range(warmup):            # exclude cold start from the stats
        time_one(url)
    samples = sorted(time_one(url) for _ in range(n))
    pct = lambda q: samples[min(int(q * len(samples)), len(samples) - 1)]
    return {
        "endpoint": path,
        "requests": n,
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(pct(0.95), 1),
        "p99_ms": round(pct(0.99), 1),
        "max_ms": round(samples[-1], 1),
        "mean_ms": round(statistics.fmean(samples), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = [bench(args.url, p, args.n)
               for p in ("/health", "/warnings", "/warnings?k=20", "/performance")]

    print(f"\n{args.url}   {args.n} requests per endpoint, 5 warm-up discarded\n")
    print(f"{'endpoint':22s}{'p50':>9s}{'p95':>9s}{'p99':>9s}{'max':>9s}")
    for r in results:
        print(f"{r['endpoint']:22s}{r['p50_ms']:>8.1f}ms{r['p95_ms']:>8.1f}ms"
              f"{r['p99_ms']:>8.1f}ms{r['max_ms']:>8.1f}ms")

    payload = {"target": args.url, "requests_per_endpoint": args.n,
               "results": results}
    if args.out:
        json.dump(payload, open(args.out, "w"), indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
