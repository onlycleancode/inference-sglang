from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_remote_infer_help() -> None:
    script = ROOT / "scripts" / "minisgl_remote_infer.py"
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "MINISGL_BASE_URL" in proc.stdout


def test_smoke_check_help() -> None:
    script = ROOT / "scripts" / "minisgl_smoke_check.py"
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "MINISGL_API_KEY" in proc.stdout


def test_lambda_chat_assets_exist() -> None:
    chat_html = ROOT / "deploy" / "lambda" / "chat.html"
    serve_script = ROOT / "scripts" / "serve_lambda_chat.py"
    assert chat_html.is_file()
    html = chat_html.read_text(encoding="utf-8")
    assert "chat/completions" in html
    assert "<!-- minisgl-chat-config -->" in html
    assert "__MINISGL_CHAT_CONFIG__" in html
    assert "/api/local/status" in html
    assert "turnOnBtn" in html
    assert serve_script.is_file()
    serve_src = serve_script.read_text(encoding="utf-8")
    assert "ENV_FILE" in serve_src
    assert "render_chat_html" in serve_src
    assert "/api/local/shutdown" in serve_src


def test_turn_on_script_exists() -> None:
    script = ROOT / "scripts" / "lambda_turn_on.sh"
    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "lambda_idle_watchdog.py" in content
    assert "serve_lambda_chat.py" in content


def test_render_chat_html_injects_api_key() -> None:
    script = ROOT / "scripts" / "serve_lambda_chat.py"
    code = f"""
import importlib.util
import os
from pathlib import Path

os.environ["MINISGL_API_KEY"] = "injected-test-key"
os.environ["MINISGL_BASE_URL"] = "http://127.0.0.1:1919/v1"
path = Path({str(script)!r})
spec = importlib.util.spec_from_file_location("serve_lambda_chat_test", path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
html = mod.render_chat_html("127.0.0.1:8765").decode()
assert "injected-test-key" in html
assert "__MINISGL_CHAT_CONFIG__" in html
assert '"baseUrl": "http://127.0.0.1:8765/v1"' in html
assert "1919/v1" not in html.split("__MINISGL_CHAT_CONFIG__", 1)[1].split("</script>", 1)[0]
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr


def test_serve_lambda_chat_help() -> None:
    script = ROOT / "scripts" / "serve_lambda_chat.py"
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "MINISGL_CHAT_LISTEN" in proc.stdout or "8765" in proc.stdout
