"""Static JavaScript/TypeScript dependency analysis worker.

This module parses import-like references from caller-supplied source text. It
never executes source code, reads arbitrary paths, installs packages, or accesses
the network.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from runtime.workers import WorkerUnavailable

MAX_FILES = 100
MAX_FILE_CHARS = 256_000
MAX_TOTAL_CHARS = 1_000_000
SUPPORTED_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}

# Static import/export-from and CommonJS require references. Dynamic imports are
# recognized only when their argument is a quoted string literal.
_IMPORT_PATTERNS = (
    re.compile(r"""\bimport\s+(?:[^;]*?\s+from\s+)?["']([^"']+)["']"""),
    re.compile(r"""\bexport\s+[^;]*?\s+from\s+["']([^"']+)["']"""),
    re.compile(r"""\brequire\(\s*["']([^"']+)["']\s*\)"""),
    re.compile(r"""\bimport\(\s*["']([^"']+)["']\s*\)"""),
)


def _safe_file_name(name: Any) -> str:
    if not isinstance(name, str) or not name or len(name) > 240:
        raise WorkerUnavailable("Each script key must be a short relative file path.")
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or (path.parts and ":" in path.parts[0]) or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerUnavailable("Script paths must be normalized relative paths without traversal.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise WorkerUnavailable("Only JavaScript/TypeScript source files are supported.")
    return path.as_posix()


def _resolve_relative(importer: str, specifier: str, files: set[str]) -> str | None:
    if not specifier.startswith("."):
        return None
    parent = PurePosixPath(importer).parent
    target = PurePosixPath(parent, specifier)
    # Normalize dot segments without allowing the path to escape the supplied
    # virtual file set. No filesystem access is performed.
    parts: list[str] = []
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    base = PurePosixPath(*parts).as_posix()
    candidates = [base]
    if not PurePosixPath(base).suffix:
        candidates.extend(base + suffix for suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))
        candidates.extend(base + "/index" + suffix for suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs"))
    for candidate in candidates:
        if candidate in files:
            return candidate
    return None


class ScriptAnalyzerWorker:
    """Trusted static-analysis worker for the GQL job registry."""

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("kind") != "script_analysis":
            raise WorkerUnavailable("Script analyzer accepts script_analysis jobs only.")
        parameters = job.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise WorkerUnavailable("Script analysis parameters must be a JSON object.")
        scripts = parameters.get("scripts")
        if not isinstance(scripts, dict) or not scripts:
            raise WorkerUnavailable("'scripts' must be a non-empty object of path-to-source strings.")
        if len(scripts) > MAX_FILES:
            raise WorkerUnavailable(f"At most {MAX_FILES} source files may be analyzed per request.")

        sources: dict[str, str] = {}
        total_chars = 0
        for raw_name, source in scripts.items():
            name = _safe_file_name(raw_name)
            if name in sources:
                raise WorkerUnavailable("Duplicate normalized script paths are not allowed.")
            if not isinstance(source, str):
                raise WorkerUnavailable("Every script source must be a string.")
            if len(source) > MAX_FILE_CHARS:
                raise WorkerUnavailable(f"Source file exceeds the {MAX_FILE_CHARS}-character limit.")
            total_chars += len(source)
            if total_chars > MAX_TOTAL_CHARS:
                raise WorkerUnavailable(f"Combined source exceeds the {MAX_TOTAL_CHARS}-character limit.")
            sources[name] = source

        file_names = set(sources)
        dependencies: dict[str, list[str]] = {name: [] for name in sources}
        external: dict[str, list[str]] = {name: [] for name in sources}
        unresolved: dict[str, list[str]] = {name: [] for name in sources}

        for name, source in sources.items():
            found: set[str] = set()
            for pattern in _IMPORT_PATTERNS:
                found.update(match.group(1) for match in pattern.finditer(source))
            for specifier in sorted(found):
                resolved = _resolve_relative(name, specifier, file_names)
                if resolved:
                    dependencies[name].append(resolved)
                elif specifier.startswith("."):
                    unresolved[name].append(specifier)
                else:
                    external[name].append(specifier)

        # Kahn ordering: dependencies precede their importers. Any residual
        # nodes are involved in a cycle or depend on a cycle.
        dependents: dict[str, list[str]] = {name: [] for name in sources}
        remaining = {name: set(items) for name, items in dependencies.items()}
        for importer, deps in dependencies.items():
            for dep in deps:
                dependents[dep].append(importer)
        ready = sorted(name for name, deps in remaining.items() if not deps)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for importer in sorted(dependents[current]):
                remaining[importer].discard(current)
                if not remaining[importer] and importer not in order and importer not in ready:
                    ready.append(importer)
            ready.sort()
        cycle_files = sorted(name for name in sources if name not in order)

        return {
            "agent": "script_analyzer_bot",
            "mode": "static_analysis_only",
            "executed_source": False,
            "file_count": len(sources),
            "dependencies": dependencies,
            "external_dependencies": external,
            "unresolved_relative_imports": unresolved,
            "execution_order": order,
            "cycle_files": cycle_files,
            "warnings": [
                "Static import parsing is heuristic and does not replace a JavaScript/TypeScript compiler.",
                "Execution order is a dependency suggestion, not permission to execute source code.",
            ],
        }
