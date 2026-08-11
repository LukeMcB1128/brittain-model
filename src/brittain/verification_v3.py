"""Local compiler and test runners for generated Brittain3 code.

These runners use timeouts and empty temporary directories. They are not a
security sandbox. Use them only with code made by trusted local models. Put the
whole process in a network-disabled container before checking third-party code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT


DEFAULT_TSC = PROJECT_ROOT / "tools" / "typescript" / "node_modules" / ".bin" / "tsc"
SUPPORTED_LANGUAGES = ("python", "typescript", "javascript", "rust", "cpp", "c", "go")


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    phase: str
    detail: str = ""


def _command_path(command: str | Path) -> str | None:
    value = str(command)
    if "/" in value:
        path = Path(value)
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


def backend_status(tsc: str | Path = DEFAULT_TSC) -> dict[str, str | None]:
    """Return the command used by each language backend, or None if absent."""
    return {
        "python": sys.executable,
        "typescript": (_command_path(tsc) if _command_path("node") else None),
        "javascript": _command_path("node"),
        "rust": _command_path("rustc"),
        "cpp": _command_path("c++"),
        "c": _command_path("cc"),
        "go": _command_path("go"),
    }


def _clean_environment(workdir: Path) -> dict[str, str]:
    keep = ("PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
    environment = {name: os.environ[name] for name in keep if name in os.environ}
    # Go requires a home and build cache. Keep both inside the disposable work
    # directory instead of exposing the user's real home directory.
    environment["HOME"] = str(workdir)
    environment["GOCACHE"] = str(workdir / ".gocache")
    return environment


def _run(command: list[str], workdir: Path, timeout: float) -> VerificationResult:
    try:
        finished = subprocess.run(
            command,
            cwd=workdir,
            env=_clean_environment(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "timeout", f"exceeded {timeout:g} seconds")
    except OSError as exc:
        return VerificationResult(False, "spawn", str(exc))
    if finished.returncode == 0:
        return VerificationResult(True, "run")
    output = (finished.stderr or finished.stdout or "nonzero exit").strip()
    return VerificationResult(False, "run", output[-2000:])


def _compile_then_run(
    compiler: list[str], executable: Path, workdir: Path, timeout: float
) -> VerificationResult:
    compiled = _run(compiler, workdir, timeout)
    if not compiled.ok:
        return VerificationResult(False, "compile", compiled.detail)
    return _run([str(executable)], workdir, timeout)


def verify_program(
    language: str,
    solution: str,
    tests: str,
    *,
    timeout: float = 10.0,
    tsc: str | Path = DEFAULT_TSC,
) -> VerificationResult:
    """Compile when necessary, then run one solution with its hidden tests."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    command = backend_status(tsc).get(language)
    if command is None:
        return VerificationResult(False, "unavailable", f"{language} toolchain is not installed")

    source = solution.rstrip() + "\n\n" + tests.rstrip() + "\n"
    with tempfile.TemporaryDirectory(prefix="brittain3-verify-") as temporary:
        workdir = Path(temporary)
        if language == "python":
            path = workdir / "candidate.py"
            path.write_text(source, encoding="utf-8")
            return _run([sys.executable, "-I", str(path)], workdir, timeout)
        if language == "javascript":
            path = workdir / "candidate.js"
            path.write_text(source, encoding="utf-8")
            return _run([str(command), str(path)], workdir, timeout)
        if language == "typescript":
            path = workdir / "candidate.ts"
            output = workdir / "out"
            path.write_text(source, encoding="utf-8")
            compiled = _run([
                str(command), str(path), "--target", "ES2022", "--module", "commonjs",
                "--strict", "--skipLibCheck", "--outDir", str(output),
            ], workdir, timeout)
            if not compiled.ok:
                return VerificationResult(False, "compile", compiled.detail)
            return _run([str(_command_path("node")), str(output / "candidate.js")], workdir, timeout)
        if language == "rust":
            path = workdir / "candidate.rs"
            executable = workdir / "candidate"
            path.write_text(source, encoding="utf-8")
            compiler = [str(command), "--edition=2021"]
            if "#[test]" in tests or "#[cfg(test)]" in tests:
                compiler.append("--test")
            compiler.extend([str(path), "-o", str(executable)])
            return _compile_then_run(compiler, executable, workdir, timeout)
        if language == "cpp":
            path = workdir / "candidate.cpp"
            executable = workdir / "candidate"
            path.write_text(source, encoding="utf-8")
            return _compile_then_run([
                str(command), "-std=c++20", "-O0", "-Wall", "-Wextra",
                str(path), "-o", str(executable),
            ], executable, workdir, timeout)
        if language == "c":
            path = workdir / "candidate.c"
            executable = workdir / "candidate"
            path.write_text(source, encoding="utf-8")
            return _compile_then_run([
                str(command), "-std=c17", "-O0", "-Wall", "-Wextra",
                str(path), "-o", str(executable),
            ], executable, workdir, timeout)

        path = workdir / "candidate.go"
        path.write_text(source, encoding="utf-8")
        return _run([str(command), "run", str(path)], workdir, timeout)
