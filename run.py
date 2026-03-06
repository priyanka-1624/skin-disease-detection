from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from training.pipeline import run_full_training


def _run_training_pipeline() -> None:
    run_full_training()


def _ensure_frontend_dependencies(frontend_dir: Path) -> None:
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        return
    subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)


def _start_servers(root: Path) -> None:
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
    frontend_cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

    popen_kwargs: dict[str, int] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    backend_proc = subprocess.Popen(backend_cmd, cwd=root, **popen_kwargs)
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=root / "frontend", **popen_kwargs)

    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:5173")

    try:
        while True:
            if backend_proc.poll() is not None:
                raise RuntimeError("Backend server terminated unexpectedly")
            if frontend_proc.poll() is not None:
                raise RuntimeError("Frontend server terminated unexpectedly")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                if os.name == "nt":
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        pass
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def main() -> None:
    root = Path(__file__).resolve().parent
    # _run_training_pipeline()  # Skip training - using existing model
    _ensure_frontend_dependencies(root / "frontend")
    _start_servers(root)


if __name__ == "__main__":
    main()
