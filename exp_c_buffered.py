
import time
from multiprocessing import Process, Queue, freeze_support


def buffered_worker(out_queue, start_idx, end_idx, batch_size=10_000):
    buffer = []

    for i in range(start_idx, end_idx):
        buffer.append(i % 7)

        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))
            buffer.clear()

    if buffer:
        out_queue.put(sum(buffer))

    # Signal that the worker has finished
    out_queue.put(None)


def run_buffered(p, total_items=10_000_000):
    q = Queue()

    # Divide the full range between workers
    chunk_size = (total_items + p - 1) // p

    workers = []

    for worker_id in range(p):
        start_idx = worker_id * chunk_size
        end_idx = min((worker_id + 1) * chunk_size, total_items)

        if start_idx < end_idx:
            worker = Process(
                target=buffered_worker,
                args=(q, start_idx, end_idx)
            )
            workers.append(worker)

    start = time.perf_counter()

    for worker in workers:
        worker.start()

    total_sum = 0
    active = len(workers)

    # Collect aggregated results
    while active > 0:
        value = q.get()

        if value is None:
            active -= 1
        else:
            total_sum += value

    for worker in workers:
        worker.join()

    elapsed = time.perf_counter() - start

    q.close()
    q.join_thread()

    return elapsed, total_sum


def main():
    cores = [1, 2, 4, 8]

    runtimes = {}
    checksums = {}

    total_items = 10_000_000

    # Independent mathematical reference
    full_cycles, remainder = divmod(total_items, 7)
    expected_checksum = full_cycles * 21 + sum(range(remainder))

    print("--- Experiment C: Buffered Aggregation ---")

    for p in cores:
        elapsed, checksum = run_buffered(p, total_items)

        runtimes[p] = elapsed
        checksums[p] = checksum

        print(
            f"P={p} | Time={elapsed:.3f}s | "
            f"Checksum={checksum}",
            flush=True
        )

    t1 = runtimes[1]

    print("\n--- Buffered Aggregation Karp-Flatt Analysis ---")

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

    print("\n--- Checksum Verification ---")
    print(f"Expected checksum: {expected_checksum}")

    if all(
        checksum == expected_checksum
        for checksum in checksums.values()
    ):
        print("CHECKSUM: PASSED")
    else:
        print("CHECKSUM: FAILED")


if __name__ == "__main__":
    freeze_support()
    main()
