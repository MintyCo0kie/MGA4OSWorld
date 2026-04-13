"""VM management via Docker SDK."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
from docker.models.containers import Container

from .client import VMClient

logger = logging.getLogger(__name__)


@dataclass
class VMConfig:
    """Configuration for VM instance."""

    # VM disk image (required)
    disk_path: str = ""  # Path to qcow2 disk image on host

    # Optional volume mounts
    logs_dir: Optional[str] = None  # Path to logs directory on host
    mount_tmp: bool = True  # Mount /tmp for file sharing

    # Docker image
    image: str = "opengui-qemu:latest"

    # QEMU options
    memory: int = 8192  # MB
    cpus: int = 4
    snapshot: bool = True  # If True, changes are discarded on shutdown

    # Container options
    container_name: Optional[str] = None
    remove_on_stop: bool = True

    # VNC port (only port we expose for debugging/viewing)
    vnc_port: Optional[int] = None  # If set, expose VNC on this host port

    # Extra QEMU args
    extra_qemu_args: str = ""

    # VM server port inside container (internal use)
    server_port: int = 5000


class VM:
    """Manages a VM instance via Docker SDK."""

    def __init__(self, config: Optional[VMConfig] = None):
        """Initialize VM with config.

        Args:
            config: VM configuration. Uses defaults if not provided.
        """
        self.config = config or VMConfig()
        self._docker_client = docker.from_env()
        self._container: Optional[Container] = None
        self._client: Optional[VMClient] = None

    @property
    def container(self) -> Optional[Container]:
        """Get Docker container if running."""
        return self._container

    @property
    def container_id(self) -> Optional[str]:
        """Get container ID if running."""
        return self._container.id if self._container else None

    @property
    def client(self) -> VMClient:
        """Get VMClient for interacting with the VM server.

        Returns:
            VMClient instance configured for this VM.

        Raises:
            RuntimeError: If VM is not running.
        """
        if self._client is None:
            raise RuntimeError("VM is not running. Call start() first.")
        return self._client

    def start(self, wait_ready: bool = True, timeout: int = 120) -> "VM":
        """Start the VM.

        Args:
            wait_ready: If True, wait for VM server to be ready.
            timeout: Maximum seconds to wait for VM to be ready.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If VM fails to start or become ready.
        """
        if self._container is not None:
            raise RuntimeError("VM is already running.")

        # Validate disk path
        if not self.config.disk_path:
            raise RuntimeError(
                "disk_path is required. Please set VMConfig.disk_path to "
                "the path of your qcow2 disk image."
            )

        disk_path = Path(self.config.disk_path).resolve()
        if not disk_path.exists():
            raise RuntimeError(f"VM disk image not found: {disk_path}")

        # Build environment variables
        env = {
            "QEMU_MEMORY": str(self.config.memory),
            "QEMU_CPUS": str(self.config.cpus),
        }

        qemu_args = []
        if self.config.snapshot:
            qemu_args.append("-snapshot")
        if self.config.extra_qemu_args:
            qemu_args.append(self.config.extra_qemu_args)
        if qemu_args:
            env["EXTRA_QEMU_ARGS"] = " ".join(qemu_args)

        # Build port bindings (only VNC if specified)
        ports = {}
        if self.config.vnc_port:
            ports["5901/tcp"] = self.config.vnc_port

        # Mount disk image into container
        volumes = {
            str(disk_path): {"bind": "/work/base.qcow2", "mode": "rw"},
        }

        # Optional: mount logs directory
        if self.config.logs_dir:
            logs_path = Path(self.config.logs_dir).resolve()
            logs_path.mkdir(parents=True, exist_ok=True)
            volumes[str(logs_path)] = {"bind": "/logs", "mode": "rw"}

        # Optional: mount /tmp for file sharing
        if self.config.mount_tmp:
            volumes["/tmp"] = {"bind": "/tmp", "mode": "rw"}

        # Container run arguments
        # Note: Don't use auto-remove so we can get logs on failure
        run_kwargs = {
            "image": self.config.image,
            "detach": True,
            "environment": env,
            "devices": ["/dev/kvm:/dev/kvm"],
            "ports": ports if ports else None,
            "volumes": volumes,
            "remove": False,  # Don't auto-remove, we handle cleanup manually
        }

        if self.config.container_name:
            run_kwargs["name"] = self.config.container_name

        logger.info(f"Starting VM with image: {self.config.image}")

        try:
            self._container = self._docker_client.containers.run(**run_kwargs)
            logger.info(f"Container started: {self._container.short_id}")
        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Docker image not found: {self.config.image}\n"
                "Please build the image first or check the image name."
            )
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to start VM: {e}") from e

        # Wait a moment for container to initialize
        time.sleep(2)

        # Check if container is still running
        try:
            self._container.reload()
            status = self._container.status
            if status != "running":
                logs = self._container.logs(tail=50).decode("utf-8", errors="replace")
                # Clean up the failed container
                try:
                    self._container.remove(force=True)
                except docker.errors.APIError:
                    pass
                self._container = None
                raise RuntimeError(
                    f"Container exited unexpectedly (status: {status}).\n"
                    f"Container logs:\n{logs}"
                )
        except docker.errors.NotFound:
            self._container = None
            raise RuntimeError(
                "Container exited immediately after starting.\n"
                "This usually means the Docker image is missing or KVM is not available.\n"
                f"Check if image exists: docker images {self.config.image}\n"
                "Check KVM access: ls -la /dev/kvm"
            )

        # Create client using docker exec
        self._client = VMClient(
            container=self._container,
            server_port=self.config.server_port,
        )

        # Wait for VM to be ready
        if wait_ready:
            logger.info("Waiting for VM server to be ready...")
            start_time = time.time()
            retries = int(timeout / 2)  # Check every 2 seconds

            if not self._client.is_ready(retries=retries, delay=2.0):
                elapsed = time.time() - start_time
                self.stop()
                raise RuntimeError(
                    f"VM server did not become ready within {elapsed:.1f}s"
                )

            logger.info("VM server is ready.")

        return self

    def stop(self) -> None:
        """Stop the VM container."""
        if self._container is None:
            logger.warning("VM is not running.")
            return

        container_id = self._container.short_id
        logger.info(f"Stopping VM: {container_id}")

        try:
            self._container.stop(timeout=10)
            logger.info("VM stopped.")
        except docker.errors.APIError as e:
            logger.error(f"Failed to stop VM: {e}")

        # Remove container if configured
        if self.config.remove_on_stop:
            try:
                self._container.remove(force=True)
                logger.info(f"Container {container_id} removed.")
            except docker.errors.APIError as e:
                logger.error(f"Failed to remove container: {e}")

        self._container = None
        self._client = None

    def kill(self) -> None:
        """Force kill the VM container."""
        if self._container is None:
            return

        try:
            self._container.kill()
        except docker.errors.APIError:
            pass
        finally:
            self._container = None
            self._client = None

    def is_running(self) -> bool:
        """Check if the VM container is running."""
        if self._container is None:
            return False

        try:
            self._container.reload()
            return self._container.status == "running"
        except docker.errors.NotFound:
            self._container = None
            self._client = None
            return False

    def logs(self, tail: int = 100) -> str:
        """Get container logs.

        Args:
            tail: Number of lines to retrieve.

        Returns:
            Container log output.
        """
        if self._container is None:
            return ""

        try:
            return self._container.logs(tail=tail).decode("utf-8", errors="replace")
        except docker.errors.APIError:
            return ""

    def exec_run(
        self,
        cmd: str,
        workdir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str]:
        """Execute a command inside the Docker container (not the VM).

        This runs in the Docker container, not inside the QEMU VM.
        For running commands inside the VM, use client.run_bash().

        Args:
            cmd: Command to run.
            workdir: Working directory.
            environment: Environment variables.

        Returns:
            Tuple of (exit_code, output).
        """
        if self._container is None:
            raise RuntimeError("VM is not running.")

        result = self._container.exec_run(
            cmd,
            workdir=workdir,
            environment=environment,
        )
        return result.exit_code, result.output.decode("utf-8", errors="replace")

    def __enter__(self) -> "VM":
        """Context manager entry - start VM."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop VM."""
        self.stop()

    # Convenience methods that delegate to client
    def health_check(self) -> Dict[str, Any]:
        """Check VM server health."""
        return self.client.health_check()

    def run_bash(self, command: str, **kwargs) -> Dict[str, Any]:
        """Run bash command in VM."""
        return self.client.run_bash(command, **kwargs)

    def run_python(self, code: str, **kwargs) -> Dict[str, Any]:
        """Run Python code in VM."""
        return self.client.run_python(code, **kwargs)

    def screenshot(self, save_path: Optional[str] = None) -> bytes:
        """Take screenshot of VM."""
        return self.client.screenshot(save_path)

    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload file to VM."""
        return self.client.upload_file(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download file from VM to host.

        Args:
            remote_path: Path to file in VM
            local_path: Path to save file on host

        Returns:
            Dict with success, remote_path, local_path, bytes (or error)
        """
        return self.client.download_file(remote_path, local_path)

    def execute_setup(
        self,
        vm_setup: Dict[str, Any],
        task_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute VM setup steps.

        Args:
            vm_setup: VM setup configuration dict with 'steps' key.
                Each step has 'action' and action-specific parameters.
            task_dir: Base directory for resolving relative file paths.
                      Required if any step uses relative paths.

        Returns:
            List of results for each step.

        Supported actions:
            - upload: Upload file to VM
                params: local (str), remote (str)
            - run_bash: Run bash command
                params: command (str), cwd (str, optional)
            - run_python: Run Python code
                params: code (str)
            - open_app: Open application (non-blocking)
                params: app (str), args (list[str], optional)
            - wait: Wait for seconds
                params: seconds (float)
        """
        steps = vm_setup.get("steps", [])
        results = []

        task_dir = Path(task_dir) if task_dir else None

        for i, step in enumerate(steps):
            action = step.get("action")
            logger.info(f"Executing setup step {i+1}/{len(steps)}: {action}")

            try:
                result = self._execute_setup_step(step, task_dir)
                results.append({"step": i, "action": action, "success": True, "result": result})
                logger.info(f"Step {i+1} completed successfully")
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                results.append({"step": i, "action": action, "success": False, "error": str(e)})
                raise RuntimeError(f"Setup step {i+1} ({action}) failed: {e}") from e

        return results

    def _execute_setup_step(
        self,
        step: Dict[str, Any],
        task_dir: Optional[Path],
    ) -> Any:
        """Execute a single setup step."""
        action = step.get("action")

        if action == "upload":
            local = step.get("local", "")
            remote = step.get("remote", "")

            # Resolve relative path for local file
            if task_dir and not Path(local).is_absolute():
                local = str(task_dir / local)

            # Upload file (goes to uploads/ directory on server)
            result = self.upload_file(local, remote)

            # The server saves files under uploads/ directory
            # We need to copy/move to the actual target location
            saved_path = result.get("saved_path", "")
            if saved_path and remote:
                # Ensure target directory exists and copy file
                remote_dir = str(Path(remote).parent)
                copy_cmd = f"mkdir -p {remote_dir} && cp '{saved_path}' '{remote}'"
                copy_result = self.run_bash(copy_cmd)
                if copy_result.get("returncode") != 0:
                    raise RuntimeError(
                        f"Failed to copy file to {remote}: {copy_result.get('stderr', '')}"
                    )
                result["final_path"] = remote

            return result

        elif action == "run_bash":
            command = step.get("command", "")
            cwd = step.get("cwd")
            return self.run_bash(command, cwd=cwd)

        elif action == "run_python":
            code = step.get("code", "")
            return self.run_python(code)

        elif action == "open_app":
            app = step.get("app", "")
            args = step.get("args", [])
            cmd = [app] + args
            return self.client.popen_start(cmd)

        elif action == "wait":
            seconds = step.get("seconds", 1)
            time.sleep(seconds)
            return {"waited": seconds}

        else:
            raise ValueError(f"Unknown action: {action}")

    def execute_eval_script(
        self,
        task_eval_script: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute task evaluation script to download files from VM.

        Similar to execute_setup, but for downloading result files after task execution.

        Args:
            task_eval_script: Evaluation script configuration dict with 'steps' key.
                Each step has 'action' and action-specific parameters.
            output_dir: Base directory to save downloaded files.
                        Required if any step uses relative paths.

        Returns:
            List of results for each step, including download success/failure info.

        Supported actions:
            - download: Download file from VM to host
                params: remote (str), local (str, optional - defaults to filename in output_dir)
            - run_bash: Run bash command in VM
                params: command (str), cwd (str, optional)
            - wait: Wait for seconds
                params: seconds (float)
        """
        steps = task_eval_script.get("steps", [])
        results = []

        output_dir = Path(output_dir) if output_dir else Path(".")

        for i, step in enumerate(steps):
            action = step.get("action")
            logger.info(f"Executing eval step {i+1}/{len(steps)}: {action}")

            try:
                result = self._execute_eval_step(step, output_dir)
                results.append({
                    "step": i,
                    "action": action,
                    "success": result.get("success", True),
                    "result": result,
                })
                if result.get("success", True):
                    logger.info(f"Eval step {i+1} completed successfully")
                else:
                    logger.warning(f"Eval step {i+1} failed: {result.get('error', 'unknown')}")
            except Exception as e:
                logger.error(f"Eval step {i+1} failed with exception: {e}")
                results.append({
                    "step": i,
                    "action": action,
                    "success": False,
                    "error": str(e),
                })
                # Don't raise - continue with other steps

        return results

    def _execute_eval_step(
        self,
        step: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Execute a single eval script step."""
        action = step.get("action")

        if action == "download":
            remote = step.get("remote", "")
            local = step.get("local", "")

            if not remote:
                return {
                    "success": False,
                    "error": "remote path is required",
                }

            # If local path not specified, use filename from remote in output_dir
            if not local:
                local = str(output_dir / Path(remote).name)
            elif not Path(local).is_absolute():
                # Resolve relative path against output_dir
                local = str(output_dir / local)

            # Download file
            result = self.download_file(remote, local)
            return result

        elif action == "run_bash":
            command = step.get("command", "")
            cwd = step.get("cwd")
            result = self.run_bash(command, cwd=cwd)
            return {
                "success": result.get("returncode") == 0,
                "result": result,
            }

        elif action == "wait":
            seconds = step.get("seconds", 1)
            time.sleep(seconds)
            return {"success": True, "waited": seconds}

        else:
            raise ValueError(f"Unknown eval action: {action}")
