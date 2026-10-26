import unittest

from fastapi.testclient import TestClient

from runtime.app import app
from runtime.jobs import clear_jobs_for_tests, get_job
from runtime.workers import WorkerRegistry, WorkerUnavailable, worker_registry


class EchoWorker:
    def execute(self, job):
        return {"model": job["model"], "accepted": True}


class FailingWorker:
    def execute(self, job):
        raise RuntimeError("private adapter detail")


class WorkerRegistryTests(unittest.TestCase):
    def test_dispatch_without_worker_fails_closed(self):
        registry = WorkerRegistry()
        with self.assertRaises(WorkerUnavailable):
            registry.dispatch({"kind": "inference"})

    def test_registered_worker_receives_job(self):
        registry = WorkerRegistry()
        registry.register("inference", EchoWorker())
        result = registry.dispatch({"kind": "inference", "model": "demo"})
        self.assertEqual(result, {"model": "demo", "accepted": True})

    def test_rejects_object_without_execute(self):
        registry = WorkerRegistry()
        with self.assertRaises(ValueError):
            registry.register("inference", object())


class DispatchEndpointTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()
        worker_registry.unregister("inference")
        self.client = TestClient(app)

    def tearDown(self):
        worker_registry.unregister("inference")
        clear_jobs_for_tests()

    def _create_job(self):
        created = self.client.post(
            "/jobs", json={"kind": "inference", "model": "demo", "parameters": {}}
        )
        self.assertEqual(created.status_code, 202)
        return created.json()["id"]

    def test_dispatch_returns_503_when_no_worker_configured(self):
        job_id = self._create_job()
        response = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(response.status_code, 503)
        self.assertIn("No trusted worker", response.json()["detail"])
        self.assertEqual(get_job(job_id)["status"], "accepted")

    def test_dispatch_returns_404_for_unknown_job(self):
        response = self.client.post("/jobs/not-a-job/dispatch")
        self.assertEqual(response.status_code, 404)

    def test_dispatch_calls_registered_worker_and_completes_job(self):
        worker_registry.register("inference", EchoWorker())
        job_id = self._create_job()
        response = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["result"]["model"], "demo")
        self.assertEqual(get_job(job_id)["status"], "completed")

        duplicate = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(duplicate.status_code, 409)

    def test_worker_failure_is_recorded_without_leaking_details(self):
        worker_registry.register("inference", FailingWorker())
        job_id = self._create_job()
        response = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Worker execution failed")
        job = get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Worker execution failed")


if __name__ == "__main__":
    unittest.main()
