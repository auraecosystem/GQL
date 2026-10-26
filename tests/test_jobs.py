import os
import tempfile
import unittest
from unittest.mock import patch

from runtime import jobs as job_registry
from runtime.jobs import (
    claim_job,
    clear_jobs_for_tests,
    create_job,
    get_job,
    list_jobs,
    update_job,
)


class JobRegistryTests(unittest.TestCase):
    def setUp(self):
        clear_jobs_for_tests()

    def tearDown(self):
        clear_jobs_for_tests()

    def test_register_job(self):
        job = create_job("inference", "example/model", {"max_tokens": 16})
        self.assertEqual(job["status"], "accepted")
        self.assertEqual(job["model"], "example/model")
        self.assertEqual(len(list_jobs()), 1)

    def test_reject_unknown_kind(self):
        with self.assertRaises(ValueError):
            create_job("destroy_cluster", "example/model")

    def test_registry_returns_copy(self):
        create_job("evaluation", "example/model")
        records = list_jobs()
        records.clear()
        self.assertEqual(len(list_jobs()), 1)

    def test_registry_retains_only_most_recent_jobs(self):
        original_limit = job_registry.MAX_JOBS
        job_registry.MAX_JOBS = 2
        try:
            create_job("inference", "model/one")
            create_job("inference", "model/two")
            create_job("inference", "model/three")
            self.assertEqual(
                [job["model"] for job in list_jobs()],
                ["model/two", "model/three"],
            )
        finally:
            job_registry.MAX_JOBS = original_limit

    def test_job_state_transitions_are_atomic(self):
        job = create_job("inference", "example/model")
        self.assertTrue(claim_job(job["id"]))
        self.assertFalse(claim_job(job["id"]))
        self.assertEqual(get_job(job["id"])["status"], "running")
        self.assertTrue(
            update_job(
                job["id"],
                expected_status="running",
                status="completed",
                result={"ok": True},
            )
        )
        self.assertFalse(
            update_job(
                job["id"],
                expected_status="running",
                status="failed",
            )
        )
        completed = get_job(job["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["result"], {"ok": True})

    def test_jobs_survive_database_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            database = os.path.join(directory, "jobs.sqlite3")
            with patch.dict(os.environ, {"GQL_JOB_DB": database}):
                job = create_job("embedding", "example/model")
                self.assertEqual(get_job(job["id"])["id"], job["id"])
                self.assertEqual(list_jobs()[0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
