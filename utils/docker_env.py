from __future__ import annotations

import concurrent.futures
import importlib
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DOCKER_WORKDIR = "/testbed"
DOCKER_USER = "root"
OFFICIAL_EVAL_IMAGE_PREFIX = "swebench/sweb.eval.x86_64."


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


def _build_logger(path: Path) -> logging.Logger:
    logger_name = f"swebench-agent-{path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def instance_id_to_image(instance_id: str) -> str:
    return f"{OFFICIAL_EVAL_IMAGE_PREFIX}{instance_id.replace('__', '_1776_')}"


def _ensure_image_present(*, client: Any, logger: logging.Logger, docker_errors: Any, image_name: str) -> None:
    try:
        client.images.get(image_name)
        logger.info("Image already exists locally, skipping pull: %s", image_name)
        return
    except docker_errors.ImageNotFound:
        logger.info("Pulling image: %s", image_name)
        client.images.pull(image_name)
        logger.info("Pulled image: %s", image_name)


class DockerEnv:
    def __init__(
        self,
        *,
        client: Any,
        container: Any,
        logger: logging.Logger,
        workdir: str = DOCKER_WORKDIR,
        user: str = DOCKER_USER,
    ) -> None:
        self.client = client
        self.container = container
        self.logger = logger
        self.workdir = workdir
        self.user = user

    @classmethod
    def create(
        cls,
        instance: dict[str, Any],
        *,
        run_id: str,
        log_dir: str | Path,
        force_rebuild: bool = False,
        max_workers: int = 1,
        namespace: str | None = None,
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
    ) -> "DockerEnv":
        docker_module = importlib.import_module("docker")
        client = docker_module.from_env()
        log_path = Path(log_dir) / "docker_env.log"
        logger = _build_logger(log_path)
        logger.info("Preparing Docker environment for %s", instance["instance_id"])
        docker_errors = docker_module.errors

        if force_rebuild or max_workers != 1 or namespace is not None or instance_image_tag != "latest" or env_image_tag != "latest":
            logger.info(
                "Ignoring build-related options; using fixed official image naming only."
            )

        image_name = instance_id_to_image(instance["instance_id"])
        _ensure_image_present(
            client=client,
            logger=logger,
            docker_errors=docker_errors,
            image_name=image_name,
        )
        container = client.containers.create(
            image=image_name,
            command=["/bin/bash", "-lc", "sleep infinity"],
            working_dir=DOCKER_WORKDIR,
            user=DOCKER_USER,
            tty=True,
            stdin_open=True,
            detach=True,
        )
        container.start()
        env = cls(client=client, container=container, logger=logger)
        pwd_result = env.run("pwd", timeout=30)
        if pwd_result.exit_code != 0 or pwd_result.stdout.strip() != DOCKER_WORKDIR:
            env.close()
            raise RuntimeError(
                f"Container workdir check failed for {instance['instance_id']}: {pwd_result.stdout!r} {pwd_result.stderr!r}"
            )
        return env

    def __enter__(self) -> "DockerEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _decode_output(self, output: Any) -> tuple[str, str]:
        if isinstance(output, tuple):
            stdout_bytes, stderr_bytes = output
        else:
            stdout_bytes, stderr_bytes = output, b""
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return stdout, stderr

    def _truncate_for_log(self, text: str, max_chars: int = 300) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}... [truncated]"

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        self.logger.info("Running command: %s", command)
        docker_command = ["/bin/bash", "-lc", command]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.container.exec_run,
                docker_command,
                workdir=self.workdir,
                user=self.user,
                demux=True,
            )
            try:
                raw_result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.logger.warning("Command timed out after %s seconds: %s", timeout, command)
                return CommandResult(command=command, stdout="", stderr=f"Timed out after {timeout} seconds", exit_code=124)

        stdout, stderr = self._decode_output(raw_result.output)
        result = CommandResult(command=command, stdout=stdout, stderr=stderr, exit_code=raw_result.exit_code)
        if result.stdout:
            self.logger.info("STDOUT: %s", self._truncate_for_log(result.stdout))
        if result.stderr:
            self.logger.info("STDERR: %s", self._truncate_for_log(result.stderr))
        self.logger.info("Exit code: %s", result.exit_code)
        return result

    def read_file(self, path: str, max_chars: int = 20000) -> str:
        result = self.run(f"cat {shlex.quote(path)}", timeout=30)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to read file {path}: {result.stderr or result.stdout}")
        return result.stdout[:max_chars]

    def find_files(self, pattern: str) -> list[str]:
        quoted_pattern = shlex.quote(pattern)
        result = self.run(f"find . -type f | grep -i -- {quoted_pattern} | head -n 200", timeout=30)
        if result.exit_code not in {0, 1}:
            raise RuntimeError(result.stderr or result.stdout)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def grep(self, text: str, path: str = ".") -> str:
        quoted_text = shlex.quote(text)
        quoted_path = shlex.quote(path)
        result = self.run(
            f"grep -RIn --exclude-dir=.git -- {quoted_text} {quoted_path} | head -n 200",
            timeout=30,
        )
        if result.exit_code not in {0, 1}:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def close(self) -> None:
        if getattr(self, "container", None) is not None:
            try:
                self.logger.info("Stopping container %s", self.container.name)
                self.container.stop(timeout=5)
            except Exception:
                pass
            try:
                self.logger.info("Removing container %s", self.container.name)
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None
