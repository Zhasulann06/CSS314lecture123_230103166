
import time
from multiprocessing import Process, Queue


def queue_worker(out_queue, items_per_worker):
    for i in range(items_per_worker):
        out_queue.put(i % 7)


def run_queue_benchmark(p, total_items=400_000):
    q = Queue()
    items_per_worker = total_items // p

    workers = [
        Process(
            target=queue_worker,
            args=(q, items_per_worker)
        )
        for _ in range(p)
    ]

    start = time.perf_counter()

    for worker in workers:
        worker.start()

    # Main process consumes all queue items
    collected = 0

    while collected < total_items:
        q.get()
        collected += 1

    for worker in workers:
        worker.join()

    elapsed = time.perf_counter() - start
    q.close()

    return elapsed


if __name__ == "__main__":
    cores = [1, 2, 4, 8]
    runtimes = {}

    for p in cores:
        elapsed = run_queue_benchmark(p)
        runtimes[p] = elapsed
        print(f"P={p} | Time={elapsed:.3f}s", flush=True)

    t1 = runtimes[1]

    print("\n--- Queue Contention Karp-Flatt Analysis ---")

    for p in [2, 4, 8]:
        tp = runtimes[p]

        speedup = t1 / tp
        efficiency = speedup / p * 100

        karp_flatt = (
            (p * tp / t1) - 1.0
        ) / (p - 1.0)

        print(
            f"P={p:2d} | "
            f"Time={tp:6.3f}s | "
            f"Speedup={speedup:5.2f}x | "
            f"Efficiency={efficiency:5.1f}% | "
            f"Karp-Flatt e={karp_flatt:6.3f}"
        )
