import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.app import app
from runtime.jobs import clear_jobs_for_tests


class GraphQLApiTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()
        self.client = TestClient(app)

    def tearDown(self):
        clear_jobs_for_tests()

    def test_health_query(self):
        response = self.client.post(
            "/graphql",
            json={"query": "{ health { status service version } }"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["health"]["status"], "ok")
        self.assertEqual(response.json()["data"]["health"]["service"], "gql-gpu-fabric")

    def test_hardware_query_maps_runtime_diagnostics(self):
        detected = {
            "platform": "test-platform",
            "python": "3.test",
            "backend_hint": "cpu_or_gpu_not_exposed",
            "rocminfo_detected": False,
            "nvidia_smi": None,
            "model_dir": "/models",
            "torch": {"available": False, "cuda_available": False, "device_count": 0},
        }
        with patch("runtime.app.detect_hardware", return_value=detected):
            response = self.client.post(
                "/graphql",
                json={"query": "{ hardware { platform backend_hint model_dir torch { available device_count } } }"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["hardware"]["platform"], "test-platform")
        self.assertEqual(response.json()["data"]["hardware"]["torch"]["device_count"], 0)

    def test_register_job_and_query_structured_parameters(self):
        response = self.client.post(
            "/graphql",
            json={
                "query": (
                    "mutation Register($input: RegisterJobInput!) { "
                    "registerJob(input: $input) { id kind model status parameters note } }"
                ),
                "variables": {
                    "input": {
                        "kind": "inference",
                        "model": "example/model",
                        "parameters": {"max_tokens": 16, "sampling": {"temperature": 0.2}},
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]["registerJob"]
        self.assertEqual(data["status"], "accepted")
        self.assertEqual(data["parameters"]["max_tokens"], 16)
        self.assertIn("Registered only", data["note"])

        listing = self.client.post(
            "/graphql",
            json={"query": "{ jobs { id model status parameters } }"},
        )
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["data"]["jobs"][0]["id"], data["id"])

    def test_invalid_job_kind_returns_graphql_error(self):
        response = self.client.post(
            "/graphql",
            json={
                "query": (
                    'mutation { registerJob(input: {kind: "destroy_cluster", '
                    'model: "example/model"}) { id } }'
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("errors", response.json())
        self.assertIn("Unsupported job kind", response.json()["errors"][0]["message"])

    def test_missing_query_is_rejected(self):
        response = self.client.post("/graphql", json={})
        self.assertEqual(response.status_code, 400)

    def test_unknown_field_is_reported_by_graphql_validation(self):
        response = self.client.post(
            "/graphql",
            json={"query": "{ health { definitelyNotAField } }"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("errors", response.json())
        self.assertIsNone(response.json()["data"])


if __name__ == "__main__":
    unittest.main()
