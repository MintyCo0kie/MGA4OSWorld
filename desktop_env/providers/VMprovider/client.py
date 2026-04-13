"""Docker exec based client for interacting with VM server."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docker.models.containers import Container

logger = logging.getLogger(__name__)


class VMClient:
    """Client for interacting with the VM's Flask server via docker exec."""

    def __init__(self, container: Container, server_port: int = 5000):
        """Initialize client with Docker container.

        Args:
            container: Docker container running the VM.
            server_port: Port of the Flask server inside the VM.
        """
        self._container = container
        self._server_port = server_port
        self._base_url = f"http://localhost:{server_port}"

    def _exec_curl(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None,
        raw_output: bool = False,
    ) -> Union[Dict, bytes]:
        """Execute curl command inside the container.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /health_check)
            json_data: JSON data to send
            timeout: Request timeout in seconds
            raw_output: If True, return raw bytes instead of parsing JSON

        Returns:
            Parsed JSON response or raw bytes if raw_output=True
        """
        cmd = ["curl", "-s", "-X", method]

        if timeout:
            cmd.extend(["--max-time", str(timeout)])

        if json_data:
            cmd.extend(["-H", "Content-Type: application/json"])
            cmd.extend(["-d", json.dumps(json_data)])

        cmd.append(f"{self._base_url}{endpoint}")

        result = self._container.exec_run(cmd)

        if result.exit_code != 0:
            raise RuntimeError(
                f"curl failed with exit code {result.exit_code}: "
                f"{result.output.decode('utf-8', errors='replace')}"
            )

        if raw_output:
            return result.output

        return json.loads(result.output)

    def _exec_curl_upload(
        self,
        endpoint: str,
        file_path: str,
        form_data: Dict[str, str],
    ) -> Dict:
        """Execute curl command with file upload.

        Args:
            endpoint: API endpoint
            file_path: Path to file inside container
            form_data: Form data to send

        Returns:
            Parsed JSON response
        """
        cmd = ["curl", "-s", "-X", "POST"]

        # Add form fields
        for key, value in form_data.items():
            cmd.extend(["-F", f"{key}={value}"])

        # Add file
        cmd.extend(["-F", f"file_data=@{file_path}"])

        cmd.append(f"{self._base_url}{endpoint}")

        result = self._container.exec_run(cmd)

        if result.exit_code != 0:
            raise RuntimeError(f"curl upload failed: {result.output.decode()}")

        return json.loads(result.output)

    # -------------------------
    # Health Check
    # -------------------------
    def health_check(self) -> Dict[str, Any]:
        """Check if the VM server is running.

        Returns:
            Dict with ok, platform, time_ms, display, session_type
        """
        return self._exec_curl("GET", "/health_check")

    def is_ready(self, retries: int = 10, delay: float = 1.0) -> bool:
        """Wait for VM server to be ready.

        Args:
            retries: Number of retries
            delay: Delay between retries in seconds

        Returns:
            True if server is ready, False otherwise
        """
        import time

        for i in range(retries):
            try:
                result = self.health_check()
                if result.get("ok"):
                    return True
            except Exception as e:
                logger.debug(f"Health check attempt {i+1}/{retries} failed: {e}")
            time.sleep(delay)
        return False

    # -------------------------
    # Subprocess: Run (short commands)
    # -------------------------
    def run_command(
        self,
        cmd: Union[str, List[str]],
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        check: bool = False,
    ) -> Dict[str, Any]:
        """Run a command and wait for it to complete.

        Args:
            cmd: Command to run
            shell: Whether to use shell execution
            cwd: Working directory
            env: Environment variables
            timeout: Command timeout in seconds
            check: If True, raise error on non-zero exit

        Returns:
            Dict with ok, returncode, stdout, stderr
        """
        payload: Dict[str, Any] = {
            "cmd": cmd,
            "shell": shell,
        }
        if cwd:
            payload["cwd"] = cwd
        if env:
            payload["env"] = env
        if timeout:
            payload["timeout"] = timeout
        if check:
            payload["check"] = check

        return self._exec_curl("POST", "/subprocess/run", json_data=payload, timeout=timeout)

    def run_shell(self, command: str, **kwargs) -> Dict[str, Any]:
        """Convenience method to run a shell command string."""
        return self.run_command(command, shell=True, **kwargs)

    def run_bash(self, command: str, **kwargs) -> Dict[str, Any]:
        """Convenience method to run command via bash -lc."""
        return self.run_command(["bash", "-lc", command], shell=False, **kwargs)

    # -------------------------
    # Subprocess: Popen (long-running commands)
    # -------------------------
    def popen_start(
        self,
        cmd: Union[str, List[str]],
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Start a long-running command.

        Returns:
            Dict with ok, job_id, pid
        """
        payload: Dict[str, Any] = {
            "cmd": cmd,
            "shell": shell,
        }
        if cwd:
            payload["cwd"] = cwd
        if env:
            payload["env"] = env

        return self._exec_curl("POST", "/subprocess/popen/start", json_data=payload)

    def popen_poll(self, job_id: str) -> Dict[str, Any]:
        """Poll a running job to check if it's still running.

        Returns:
            Dict with ok, job_id, pid, running, returncode
        """
        return self._exec_curl("GET", f"/subprocess/popen/{job_id}/poll")

    def popen_wait(self, job_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Wait for a job to complete and get its output.

        Returns:
            Dict with ok, job_id, pid, returncode, stdout, stderr
        """
        payload = {}
        if timeout:
            payload["timeout"] = timeout

        return self._exec_curl(
            "POST",
            f"/subprocess/popen/{job_id}/wait",
            json_data=payload if payload else None,
            timeout=timeout,
        )

    def popen_terminate(self, job_id: str, signal: str = "TERM") -> Dict[str, Any]:
        """Terminate a running job.

        Args:
            job_id: Job ID
            signal: Signal to send ("TERM" or "KILL")

        Returns:
            Dict with ok, job_id, signal
        """
        payload = {"sig": signal}
        return self._exec_curl("POST", f"/subprocess/popen/{job_id}/terminate", json_data=payload)

    # -------------------------
    # Run Python
    # -------------------------
    def run_python(
        self,
        code: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Python code in the VM.

        Returns:
            Dict with ok, returncode, stdout, stderr
        """
        payload: Dict[str, Any] = {"code": code}
        if cwd:
            payload["cwd"] = cwd
        if env:
            payload["env"] = env
        if timeout:
            payload["timeout"] = timeout

        return self._exec_curl("POST", "/run_python", json_data=payload, timeout=timeout)

    # -------------------------
    # File Upload
    # -------------------------
    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload a file to the VM.

        Uses /tmp as shared volume to transfer file into container,
        then uploads to VM via the server API.

        Args:
            local_path: Path to local file on host
            remote_path: Desired path in VM

        Returns:
            Dict with ok, saved_path, bytes
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Copy file to /tmp (shared volume)
        tmp_path = f"/tmp/{local_path.name}"
        with open(local_path, "rb") as f:
            file_data = f.read()

        # Write to /tmp in container
        # Use base64 to safely transfer binary data
        import base64
        b64_data = base64.b64encode(file_data).decode()

        write_cmd = f"echo '{b64_data}' | base64 -d > {tmp_path}"
        result = self._container.exec_run(["bash", "-c", write_cmd])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to write file to container: {result.output.decode()}")

        # Upload from container to VM
        return self._exec_curl_upload(
            "/setup/upload",
            tmp_path,
            {"file_path": remote_path},
        )

    # -------------------------
    # File Download
    # -------------------------
    def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> Dict[str, Any]:
        """Download a file from the VM to the host.

        Args:
            remote_path: Path to file in VM
            local_path: Path to save file on host

        Returns:
            Dict with success, remote_path, local_path, bytes, error (if failed)
        """
        import urllib.parse

        # URL encode the remote path
        encoded_path = urllib.parse.quote(remote_path, safe='')

        try:
            # Download file from VM server
            file_bytes = self._exec_curl(
                "GET",
                f"/setup/download?file_path={encoded_path}",
                raw_output=True,
            )

            # Check if the response is an error message (JSON)
            # The server returns file content directly on success, JSON on error
            try:
                # Try to parse as JSON - if it works, it's an error
                error_response = json.loads(file_bytes)
                return {
                    "success": False,
                    "remote_path": remote_path,
                    "local_path": local_path,
                    "error": error_response.get("error", "Unknown error"),
                }
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON, it's actual file content
                pass

            # Save to local path
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_bytes)

            return {
                "success": True,
                "remote_path": remote_path,
                "local_path": str(local_path),
                "bytes": len(file_bytes),
            }

        except RuntimeError as e:
            error_msg = str(e)
            # Parse error from curl output
            if "404" in error_msg:
                error_msg = "File not found"
            elif "403" in error_msg:
                error_msg = "File not readable"
            elif "400" in error_msg:
                error_msg = "Not a regular file"

            return {
                "success": False,
                "remote_path": remote_path,
                "local_path": local_path,
                "error": error_msg,
            }

    # -------------------------
    # Screenshot
    # -------------------------
    def screenshot(self, save_path: Optional[str] = None) -> bytes:
        """Take a screenshot of the VM.

        Args:
            save_path: Optional path to save the screenshot on host

        Returns:
            Screenshot bytes (PNG)
        """
        screenshot_bytes = self._exec_curl("GET", "/screenshot", raw_output=True)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(screenshot_bytes)

        return screenshot_bytes
