from __future__ import annotations

from qiskit.providers import JobStatus, JobV1


class RocQuantumJob(JobV1):
    """Synchronous Qiskit job wrapper for local rocQuantum simulation."""

    def __init__(self, backend, job_id: str, result):
        super().__init__(backend, job_id)
        self._result = result
        self._status = JobStatus.DONE

    def submit(self):
        raise RuntimeError("RocQuantumJob is executed synchronously by backend.run().")

    def result(self, timeout=None):
        return self._result

    def status(self):
        return self._status

    def cancel(self):
        return False

    def cancelled(self):
        return False

    def done(self):
        return self._status == JobStatus.DONE

    def running(self):
        return False
