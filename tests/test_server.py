import pytest

from venusocr import cli, server


def test_serve_options_and_vllm_passthrough(monkeypatch):
    commands = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "venusocr",
            "serve",
            "./model",
            "--host",
            "127.0.0.1",
            "--port",
            "18000",
            "--",
            "--max-model-len",
            "4096",
        ],
    )
    monkeypatch.setattr(server, "launch_server", lambda command: commands.append(command) or 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    command = commands[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18000"
    assert command[-2:] == ["--max-model-len", "4096"]
    assert "--" not in command
    assert command.count("--port") == 1


def test_server_limits_default_plugins(monkeypatch):
    environments = []
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(server.shutil, "which", lambda executable: "/bin/vllm")
    monkeypatch.setattr(
        server.subprocess,
        "call",
        lambda command, env: environments.append(env) or 0,
    )
    assert server.launch_server(["vllm", "serve", "./model"]) == 0
    assert environments[0]["VLLM_PLUGINS"] == "venusocr"
