from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from utils.docker_env import DockerEnv


class FakeContainer:
    def __init__(self) -> None:
        self.name = "fake-container"

    def exec_run(self, command, workdir=None, user=None, demux=False):
        joined = " ".join(command) if isinstance(command, list) else command
        if "cat" in joined:
            output = (b"hello world", b"")
        elif "grep -RIn" in joined:
            output = (b"./a.py:1:needle\n", b"")
        elif "find . -type f" in joined:
            output = (b"./README.md\n./src/a.py\n", b"")
        else:
            output = (b"/testbed\n", b"")
        return SimpleNamespace(exit_code=0, output=output)

    def stop(self, timeout=5):
        return None

    def remove(self, force=True):
        return None


class DockerEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DockerEnv(client=None, container=FakeContainer(), logger=logging.getLogger("test-docker"))

    def test_run_executes_in_container(self) -> None:
        result = self.env.run("pwd")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("/testbed", result.stdout)

    def test_read_file_returns_contents(self) -> None:
        self.assertEqual(self.env.read_file("README.md"), "hello world")

    def test_find_files_parses_lines(self) -> None:
        self.assertEqual(self.env.find_files("README"), ["./README.md", "./src/a.py"])

    def test_grep_returns_stdout(self) -> None:
        self.assertIn("needle", self.env.grep("needle"))


if __name__ == "__main__":
    unittest.main()
