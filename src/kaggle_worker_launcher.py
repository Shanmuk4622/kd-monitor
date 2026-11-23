"""Run each Kaggle stage with the software recorded by the completed preflight."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


QUALIFIED_SOFTWARE = {
    "python": "3.12.13", "torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128",
    "cuda": "12.8", "cudnn": 91002, "numpy": "2.0.2", "pillow": "11.3.0", "timm": "1.0.26",
}


def compatible_kernel():
    if sys.version.split()[0] != QUALIFIED_SOFTWARE["python"]:
        return False
    try:
        from kaggle_run import runtime_id
        actual = runtime_id()
    except (ImportError, OSError, RuntimeError):
        return False
    return all(actual.get(k) == v for k, v in QUALIFIED_SOFTWARE.items())


def install_worker_python(root, env):
    root.mkdir(parents=True, exist_ok=True)
    tools = root / "tools"
    uv = tools / "bin" / "uv"
    if not uv.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade",
                        "--target", str(tools), "uv"], check=True)
    env.update(UV_PYTHON_INSTALL_DIR=str(root / "interpreters"), UV_CACHE_DIR=str(root / "cache"),
               UV_LINK_MODE="hardlink")
    venv = root / "venv"
    python = venv / "bin" / "python"
    if not python.exists():
        print("Installing Python 3.12.13 in Kaggle scratch space.", flush=True)
        subprocess.run([uv, "venv", "--managed-python", "--python", QUALIFIED_SOFTWARE["python"], venv],
                       env=env, check=True)
    wheels = []
    for package in ("torch", "torchvision"):
        version = QUALIFIED_SOFTWARE[package].replace("+", "%2B")
        wheel = f"{package}-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
        wheels.append(f"https://download.pytorch.org/whl/cu128/{wheel}")
    requirements = [*wheels, "numpy==2.0.2", "pillow==11.3.0", "timm==1.0.26",
                    "huggingface_hub", "requests", "scipy", "filelock", "nvidia-ml-py", "psutil"]
    print("Checking the preflight packages and installing any missing CUDA 12.8 wheels.", flush=True)
    subprocess.run([uv, "pip", "install", "--python", python, *requirements], env=env, check=True)
    return python


def launch_stage(runtime, program, settings, role):
    settings = {**settings, "expected_software": QUALIFIED_SOFTWARE}
    if compatible_kernel():
        exec(compile(program, f"<{role}>", "exec"), dict(settings))
        return
    print("Preparing the preflight Python 3.12.13 / CUDA 12.8 environment. "
          "The first run downloads the GPU packages.", flush=True)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    # Keep the mounted driver visible. Load CUDA user libraries from the qualified wheels.
    libs = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in libs if p and "site-packages" not in p
                                              and not ("/cuda" in p and not p.endswith("/compat")))
    root = Path("/kaggle/temp") / f"kd-{role}-python31213-cu128"
    python = install_worker_python(root, env)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        # Share only Kaggle's Python secrets client. It obtains credentials in the child.
        for module in ("kaggle_web_client", "kaggle_secrets"):
            spec = importlib.util.find_spec(module)
            if not spec or not spec.origin or not spec.origin.endswith(".py"):
                raise RuntimeError(f"Kaggle's {module} module is unavailable in this session.")
            shutil.copyfile(spec.origin, folder / (module + ".py"))
        script = folder / "stage.py"
        script.write_text("import sys, json\n"
                          + f"sys.path.insert(0, {str(runtime)!r})\n"
                          + "globals().update(json.loads(sys.argv[1]))\n" + program, encoding="utf-8")
        # Settings contain paths and worker options. Tokens are read through Kaggle Secrets.
        command = [str(python), "-u", str(script), json.dumps(settings)]
        print(f"Starting {role} in the qualified environment.", flush=True)
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", start_new_session=True) as proc:
            def forward(signum, frame):
                if proc.poll() is None:
                    proc.send_signal(signum)
            previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                if proc.wait():
                    raise RuntimeError(f"{role} exited with status {proc.returncode}. See its output above.")
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
