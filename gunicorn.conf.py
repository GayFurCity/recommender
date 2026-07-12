import os


def child_exit(server, worker):
    # Cleans up the per-worker metric files prometheus_client writes under
    # PROMETHEUS_MULTIPROC_DIR, so they don't pile up as gunicorn recycles workers.
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess
        multiprocess.mark_process_dead(worker.pid)
