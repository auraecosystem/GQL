import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.app import app
from runtime.jobs import clear_jobs_for_tests
from runtime.workers import worker_registry


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()
        self.client = TestClient(app)

    def tearDown(self):
        clear_jobs_for_tests()

    def test_inference_endpoint_is_disabled_without_token_configuration(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": ""}):
            response = self.client.post(
                "/v1/inference", json={"kind": "inference", "model": "demo"}
            )
        self.assertEqual(response.status_code, 503)

    def test_inference_endpoint_requires_bearer_token(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
            response = self.client.post(
                "/v1/inference", json={"kind": "inference", "model": "demo"}
            )
        self.assertEqual(response.status_code, 401)

    def test_inference_endpoint_rejects_non_inference_kind(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
            response = self.client.post(
                "/v1/inference",
                headers={"Authorization": "Bearer test-secret"},
                json={"kind": "training", "model": "demo"},
            )
        self.assertEqual(response.status_code, 422)

    def test_authenticated_inference_executes_registered_worker(self):
        class EchoWorker:
            def execute(self, job):
                return {"model": job["model"], "detection_count": 0}

        worker_registry.unregister("inference")
        worker_registry.register("inference", EchoWorker())
        try:
            with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
                response = self.client.post(
                    "/v1/inference",
                    headers={"Authorization": "Bearer test-secret"},
                    json={"kind": "inference", "model": "demo"},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "completed")
            self.assertEqual(response.json()["result"]["model"], "demo")
        finally:
            worker_registry.unregister("inference")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_job_submission_listing_and_detail(self):
        response = self.client.post(
            "/jobs",
            json={"kind": "inference", "model": "example/model", "parameters": {"max_tokens": 8}},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "accepted")
        job_id = response.json()["id"]

        listing = self.client.get("/jobs")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["count"], 1)

        detail = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["model"], "example/model")
        self.assertEqual(detail.json()["status"], "accepted")

    def test_unknown_job_kind_is_rejected(self):
        response = self.client.post("/jobs", json={"kind": "destroy_cluster", "model": "example/model"})
        self.assertEqual(response.status_code, 422)

    def test_missing_model_is_rejected(self):
        response = self.client.post("/jobs", json={"kind": "inference"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_job_detail_returns_404(self):
        response = self.client.get("/jobs/not-a-job")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
