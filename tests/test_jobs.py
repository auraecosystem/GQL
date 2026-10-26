import unittest
from runtime.jobs import clear_jobs_for_tests, create_job, list_jobs


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


if __name__ == "__main__":
    unittest.main()
