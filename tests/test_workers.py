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
        with self.assertRaises(WorkerUnavailable):
            WorkerRegistry().dispatch({"kind": "inference"})

    def test_registered_worker_receives_job(self):
        registry = WorkerRegistry()
        registry.register("inference", EchoWorker())
        self.assertEqual(
            registry.dispatch({"kind": "inference", "model": "demo"}),
            {"model": "demo", "accepted": True},
        )

    def test_rejects_object_without_execute(self):
        with self.assertRaises(ValueError):
            WorkerRegistry().register("inference", object())

    def test_multiple_workers_coexist_and_require_explicit_selection(self):
        registry = WorkerRegistry()
        registry.register("inference", EchoWorker(), name="echo")
        registry.register("inference", FailingWorker(), name="failing")
        self.assertEqual(registry.available("inference"), ["echo", "failing"])
        with self.assertRaisesRegex(WorkerUnavailable, "Set parameters.worker"):
            registry.dispatch({"kind": "inference", "parameters": {}})
        result = registry.dispatch({
            "kind": "inference", "model": "demo",
            "parameters": {"worker": "echo"},
        })
        self.assertTrue(result["accepted"])

    def test_unknown_worker_selection_fails_closed(self):
        registry = WorkerRegistry()
        registry.register("inference", EchoWorker(), name="echo")
        with self.assertRaisesRegex(WorkerUnavailable, "not configured"):
            registry.dispatch({"kind": "inference", "parameters": {"worker": "missing"}})

    def test_unregister_one_named_worker_preserves_other_workers(self):
        registry = WorkerRegistry()
        registry.register("inference", EchoWorker(), name="echo")
        registry.register("inference", FailingWorker(), name="failing")
        registry.unregister("inference", "failing")
        self.assertEqual(registry.available("inference"), ["echo"])


class DispatchEndpointTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()
        worker_registry.unregister("inference")
        self.client = TestClient(app)

    def tearDown(self):
        worker_registry.unregister("inference")
        clear_jobs_for_tests()

    def _create_job(self, parameters=None):
        created = self.client.post(
            "/jobs", json={"kind": "inference", "model": "demo", "parameters": parameters or {}}
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
        self.assertEqual(self.client.post(f"/jobs/{job_id}/dispatch").status_code, 409)

    def test_multiple_registered_workers_can_be_selected_through_api(self):
        worker_registry.register("inference", EchoWorker(), name="echo")
        worker_registry.register("inference", FailingWorker(), name="failing")
        job_id = self._create_job({"worker": "echo"})
        response = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["accepted"], True)

    def test_ambiguous_worker_selection_returns_503_and_keeps_job_accepted(self):
        worker_registry.register("inference", EchoWorker(), name="echo")
        worker_registry.register("inference", FailingWorker(), name="failing")
        job_id = self._create_job()
        response = self.client.post(f"/jobs/{job_id}/dispatch")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Multiple workers", response.json()["detail"])
        self.assertEqual(get_job(job_id)["status"], "accepted")

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
