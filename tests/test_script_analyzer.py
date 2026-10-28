import unittest

from runtime.script_analyzer import ScriptAnalyzerWorker
from runtime.workers import WorkerUnavailable


class ScriptAnalyzerWorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = ScriptAnalyzerWorker()

    def analyze(self, scripts):
        return self.worker.execute({
            "kind": "script_analysis",
            "model": "script-analyzer",
            "parameters": {"scripts": scripts},
        })

    def test_dependency_order_places_dependency_first(self):
        result = self.analyze({
            "src/main.js": 'import { run } from "./lib.js"; run();',
            "src/lib.js": "export function run() { return 1; }",
        })
        self.assertEqual(result["agent"], "script_analyzer_bot")
        self.assertFalse(result["executed_source"])
        self.assertEqual(result["execution_order"], ["src/lib.js", "src/main.js"])
        self.assertEqual(result["dependencies"]["src/main.js"], ["src/lib.js"])

    def test_detects_cycles(self):
        result = self.analyze({
            "a.js": 'import "./b.js";',
            "b.js": 'import "./a.js";',
        })
        self.assertEqual(result["cycle_files"], ["a.js", "b.js"])
        self.assertEqual(result["execution_order"], [])

    def test_separates_external_and_unresolved_imports(self):
        result = self.analyze({
            "main.ts": 'import React from "react"; import "./missing.js";',
        })
        self.assertEqual(result["external_dependencies"]["main.ts"], ["react"])
        self.assertEqual(result["unresolved_relative_imports"]["main.ts"], ["./missing.js"])

    def test_rejects_path_traversal(self):
        with self.assertRaises(WorkerUnavailable):
            self.analyze({"../outside.js": "export {};"} )

    def test_rejects_absolute_paths(self):
        with self.assertRaises(WorkerUnavailable):
            self.analyze({"/etc/evil.js": "export {};"} )

    def test_rejects_non_source_extensions(self):
        with self.assertRaises(WorkerUnavailable):
            self.analyze({"script.py": "print('not executed')"} )

    def test_rejects_missing_scripts(self):
        with self.assertRaises(WorkerUnavailable):
            self.analyze({})

    def test_rejects_non_string_source(self):
        with self.assertRaises(WorkerUnavailable):
            self.analyze({"main.js": {"code": "x"}})

    def test_rejects_wrong_job_kind(self):
        with self.assertRaises(WorkerUnavailable):
            self.worker.execute({"kind": "inference", "parameters": {}})

    def test_does_not_execute_supplied_source(self):
        result = self.analyze({"main.js": 'process.exit(99); import "./dep.js";', "dep.js": "export {};"} )
        self.assertFalse(result["executed_source"])
        self.assertIn("main.js", result["execution_order"])


if __name__ == "__main__":
    unittest.main()
