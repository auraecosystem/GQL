import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.app import app
from runtime.jobs import clear_jobs_for_tests


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()
        self.client = TestClient(app)

    def tearDown(self):
        clear_jobs_for_tests()

    def test_inference_api_disabled_without_token(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": ""}):
            response = self.client.post(
                "/v1/inference", json={"kind": "inference", "model": "demo"}
            )
        self.assertEqual(response.status_code, 503)

    def test_inference_api_requires_bearer_token(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
            response = self.client.post(
                "/v1/inference", json={"kind": "inference", "model": "demo"}
            )
        self.assertEqual(response.status_code, 401)

    def test_inference_api_rejects_non_inference_jobs(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
            response = self.client.post(
                "/v1/inference",
                headers={"Authorization": "Bearer test-secret"},
                json={"kind": "training", "model": "demo"},
            )
        self.assertEqual(response.status_code, 422)

    def test_authenticated_inference_dispatches_trusted_worker(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}), patch(
            "runtime.app.worker_registry.dispatch", return_value={"model": "demo", "ok": True}
        ):
            response = self.client.post(
                "/v1/inference",
                headers={"Authorization": "Bearer test-secret"},
                json={"kind": "inference", "model": "demo"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["result"], {"model": "demo", "ok": True})

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


    def test_script_analyzer_requires_bearer_token(self):
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}):
            response = self.client.post(
                "/v1/agents/script-analyzer",
                json={"scripts": {"main.js": "export {};"}},
            )
        self.assertEqual(response.status_code, 401)

    def test_authenticated_script_analyzer_dispatches_registered_worker(self):
        expected = {"agent": "script_analyzer_bot", "executed_source": False, "execution_order": ["main.js"]}
        with patch.dict(os.environ, {"GQL_API_TOKEN": "test-secret"}), patch(
            "runtime.app.worker_registry.dispatch", return_value=expected
        ) as dispatch:
            response = self.client.post(
                "/v1/agents/script-analyzer",
                headers={"Authorization": "Bearer test-secret"},
                json={"scripts": {"main.js": "export {};"}},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        dispatch.assert_called_once()

if __name__ == "__main__":
    unittest.main()
