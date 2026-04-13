"""
Action module for MGA Agent.
Supports two execution modes:
1. PyAutoGUI mode: GUI operations (click, type, scroll, etc.)
2. Code mode: Direct code execution (shell commands, Python scripts, etc.)

Supports both local execution (subprocess) and remote execution (env_controller).
"""

import ast
import logging
import re
import subprocess
import sys
import tempfile
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("desktopenv.agent.actions")


class ActionType(Enum):
    """Types of actions the agent can perform."""
    # PyAutoGUI actions
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    PRESS = "press"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    MOVE = "move"
    
    # Code execution actions
    SHELL = "shell"
    PYTHON = "python"
    BASH = "bash"
    
    # Special actions
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    DONE = "done"
    FAIL = "fail"


class ExecutionMode(Enum):
    """Execution mode for the agent."""
    PYAUTOGUI = "pyautogui"  # GUI operations
    CODE = "code"            # Direct code execution
    HYBRID = "hybrid"        # Both modes available


class ExecutionTarget(Enum):
    """Where to execute the action."""
    LOCAL = "local"          # Execute on host machine
    REMOTE = "remote"        # Execute on remote VM via env_controller


@dataclass
class Action:
    """Represents a single action to be executed."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_code: str = ""
    description: str = ""
    
    def to_pyautogui_code(self) -> str:
        """Convert action to PyAutoGUI code string."""
        if self.action_type == ActionType.CLICK:
            x, y = self.parameters.get("x", 0), self.parameters.get("y", 0)
            return f"pyautogui.click({x}, {y})"
        elif self.action_type == ActionType.DOUBLE_CLICK:
            x, y = self.parameters.get("x", 0), self.parameters.get("y", 0)
            return f"pyautogui.doubleClick({x}, {y})"
        elif self.action_type == ActionType.RIGHT_CLICK:
            x, y = self.parameters.get("x", 0), self.parameters.get("y", 0)
            return f"pyautogui.rightClick({x}, {y})"
        elif self.action_type == ActionType.TYPE:
            text = self.parameters.get("text", "")
            return f"pyautogui.write({repr(text)})"
        elif self.action_type == ActionType.PRESS:
            key = self.parameters.get("key", "")
            return f"pyautogui.press({repr(key)})"
        elif self.action_type == ActionType.HOTKEY:
            keys = self.parameters.get("keys", [])
            keys_str = ", ".join(repr(k) for k in keys)
            return f"pyautogui.hotkey({keys_str})"
        elif self.action_type == ActionType.SCROLL:
            amount = self.parameters.get("amount", 0)
            x = self.parameters.get("x")
            y = self.parameters.get("y")
            if x is not None and y is not None:
                return f"pyautogui.scroll({amount}, x={x}, y={y})"
            return f"pyautogui.scroll({amount})"
        elif self.action_type == ActionType.DRAG:
            x1, y1 = self.parameters.get("start_x", 0), self.parameters.get("start_y", 0)
            x2, y2 = self.parameters.get("end_x", 0), self.parameters.get("end_y", 0)
            return f"pyautogui.moveTo({x1}, {y1})\npyautogui.drag({x2-x1}, {y2-y1})"
        elif self.action_type == ActionType.MOVE:
            x, y = self.parameters.get("x", 0), self.parameters.get("y", 0)
            return f"pyautogui.moveTo({x}, {y})"
        elif self.action_type == ActionType.WAIT:
            seconds = self.parameters.get("seconds", 1)
            return f"time.sleep({seconds})"
        elif self.action_type == ActionType.DONE:
            return "# Task completed"
        elif self.action_type == ActionType.FAIL:
            reason = self.parameters.get("reason", "Unknown error")
            return f"# Task failed: {reason}"
        else:
            return self.raw_code
    
    def to_shell_code(self) -> str:
        """Get shell command from action."""
        if self.action_type == ActionType.SHELL:
            return self.parameters.get("command", self.raw_code)
        elif self.action_type == ActionType.BASH:
            return self.parameters.get("script", self.raw_code)
        return self.raw_code
    
    def to_python_code(self) -> str:
        """Get Python code from action."""
        if self.action_type == ActionType.PYTHON:
            return self.parameters.get("code", self.raw_code)
        return self.raw_code


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    output: str = ""
    error: str = ""
    return_code: int = 0
    screenshot: Optional[bytes] = None
    
    @classmethod
    def from_env_controller_result(cls, result: Dict) -> "ActionResult":
        """Create ActionResult from env_controller response format."""
        status = result.get("status", "unknown")
        return cls(
            success=status == "success",
            output=result.get("output", ""),
            error=result.get("error", ""),
            return_code=result.get("returncode", result.get("return_code", -1)),
        )


class ActionExecutorBase(ABC):
    """Abstract base class for action executors."""
    
    @abstractmethod
    def execute(self, action: Action) -> ActionResult:
        """Execute a single action."""
        pass
    
    def execute_batch(self, actions: List[Action]) -> List[ActionResult]:
        """Execute a batch of actions."""
        results = []
        for action in actions:
            result = self.execute(action)
            results.append(result)
            if action.action_type == ActionType.FAIL or not result.success:
                break  # Stop on failure
        return results


class PyAutoGUIExecutor(ActionExecutorBase):
    """
    Executor for PyAutoGUI-based GUI operations.
    Supports both local execution and remote execution via env_controller.
    """
    
    def __init__(
        self,
        platform: str = "linux",
        width: int = 1920,
        height: int = 1080,
        env_controller = None,  # For remote execution
    ):
        self.platform = platform
        self.width = width
        self.height = height
        self.env_controller = env_controller
    
    def execute(self, action: Action) -> ActionResult:
        """Execute a single PyAutoGUI action."""
        try:
            code = action.to_pyautogui_code()
            if not code or code.startswith("#"):
                return ActionResult(success=True, output=code)
            
            # Wrap code with imports
            full_code = f"""
import pyautogui
import time
pyautogui.FAILSAFE = False
{code}
"""
            
            if self.env_controller:
                return self._execute_remote(full_code)
            else:
                return self._execute_local(full_code)
                
        except Exception as e:
            logger.error(f"PyAutoGUI execution error: {e}")
            return ActionResult(success=False, error=str(e))
    
    def _execute_local(self, code: str) -> ActionResult:
        """Execute PyAutoGUI code locally."""
        try:
            import pyautogui
            import time
            
            exec_globals = {"pyautogui": pyautogui, "time": time}
            exec(code, exec_globals)
            return ActionResult(success=True, output="Executed successfully")
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def _execute_remote(self, code: str) -> ActionResult:
        """Execute PyAutoGUI code on remote VM via env_controller."""
        try:
            result = self.env_controller.run_python_script(code)
            return ActionResult.from_env_controller_result(result)
        except Exception as e:
            return ActionResult(success=False, error=str(e))


class CodeExecutor(ActionExecutorBase):
    """
    Executor for direct code execution (shell, Python, etc.).
    Supports both local execution (subprocess) and remote execution (env_controller).
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        safe_mode: bool = True,
        env_controller = None,  # For remote execution on VMware
    ):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
        self.safe_mode = safe_mode
        self.env_controller = env_controller
        
        # Dangerous commands to block in safe mode
        self.blocked_commands = [
            "rm -rf /",
            "mkfs",
            ":(){:|:&};:",
            "dd if=/dev/zero",
            "chmod -R 777 /",
        ]
    
    def execute(self, action: Action) -> ActionResult:
        """Execute a code action."""
        try:
            if action.action_type == ActionType.SHELL:
                command = action.parameters.get("command", action.raw_code)
                return self._execute_shell(command)
            elif action.action_type == ActionType.BASH:
                script = action.parameters.get("script", action.raw_code)
                return self._execute_bash(script)
            elif action.action_type == ActionType.PYTHON:
                code = action.parameters.get("code", action.raw_code)
                return self._execute_python(code)
            elif action.action_type == ActionType.DONE:
                return ActionResult(success=True, output="Task completed")
            elif action.action_type == ActionType.FAIL:
                reason = action.parameters.get("reason", "Task failed")
                return ActionResult(success=False, error=reason)
            else:
                return ActionResult(
                    success=False, 
                    error=f"Unsupported action type: {action.action_type}"
                )
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return ActionResult(success=False, error=str(e))
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute."""
        if not self.safe_mode:
            return True
        
        command_lower = command.lower()
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return False
        return True
    
    def _execute_shell(self, command: str) -> ActionResult:
        """Execute a shell command."""
        if not self._is_safe_command(command):
            return ActionResult(success=False, error="Command blocked for safety")
        
        # Use env_controller for remote execution
        if self.env_controller:
            return self._execute_remote_bash(command)
        
        # Local execution
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ActionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, error="Command timed out")
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def _execute_bash(self, script: str) -> ActionResult:
        """Execute a bash script."""
        if not self._is_safe_command(script):
            return ActionResult(success=False, error="Script blocked for safety")
        
        # Use env_controller for remote execution
        if self.env_controller:
            return self._execute_remote_bash(script)
        
        # Local execution
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as f:
                f.write("#!/bin/bash\n")
                f.write(script)
                script_path = f.name
            
            os.chmod(script_path, 0o755)
            
            result = subprocess.run(
                ["bash", script_path],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            os.unlink(script_path)
            
            return ActionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, error="Script timed out")
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def _execute_python(self, code: str) -> ActionResult:
        """Execute Python code."""
        # Use env_controller for remote execution
        if self.env_controller:
            return self._execute_remote_python(code)
        
        # Local execution
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                script_path = f.name
            
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            os.unlink(script_path)
            
            return ActionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, error="Python script timed out")
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def _execute_remote_bash(self, script: str) -> ActionResult:
        """Execute bash script on remote VM via env_controller."""
        try:
            result = self.env_controller.run_bash_script(script, timeout=self.timeout)
            return ActionResult.from_env_controller_result(result)
        except Exception as e:
            logger.error(f"Remote bash execution error: {e}")
            return ActionResult(success=False, error=str(e))
    
    def _execute_remote_python(self, code: str) -> ActionResult:
        """Execute Python code on remote VM via env_controller."""
        try:
            result = self.env_controller.run_python_script(code)
            return ActionResult.from_env_controller_result(result)
        except Exception as e:
            logger.error(f"Remote Python execution error: {e}")
            return ActionResult(success=False, error=str(e))


class HybridExecutor(ActionExecutorBase):
    """
    Hybrid executor that can handle both PyAutoGUI and code actions.
    Automatically selects the appropriate executor based on action type.
    Supports both local and remote execution.
    """
    
    def __init__(
        self,
        platform: str = "linux",
        width: int = 1920,
        height: int = 1080,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        safe_mode: bool = True,
        env_controller = None,  # For remote execution on VMware
    ):
        self.env_controller = env_controller
        
        self.pyautogui_executor = PyAutoGUIExecutor(
            platform=platform,
            width=width,
            height=height,
            env_controller=env_controller,
        )
        self.code_executor = CodeExecutor(
            working_dir=working_dir,
            timeout=timeout,
            safe_mode=safe_mode,
            env_controller=env_controller,
        )
    
    def execute(self, action: Action) -> ActionResult:
        """Execute action using appropriate executor."""
        if action.action_type in [
            ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK,
            ActionType.TYPE, ActionType.PRESS, ActionType.HOTKEY,
            ActionType.SCROLL, ActionType.DRAG, ActionType.MOVE,
            ActionType.WAIT, ActionType.SCREENSHOT,
        ]:
            return self.pyautogui_executor.execute(action)
        elif action.action_type in [
            ActionType.SHELL, ActionType.BASH, ActionType.PYTHON,
        ]:
            return self.code_executor.execute(action)
        elif action.action_type in [ActionType.DONE, ActionType.FAIL]:
            return self.code_executor.execute(action)
        else:
            return ActionResult(
                success=False, 
                error=f"Unknown action type: {action.action_type}"
            )


class ActionParser:
    """Parse LLM responses into Action objects."""
    
    # Patterns for different action formats
    PYAUTOGUI_PATTERNS = {
        "click": re.compile(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
        "double_click": re.compile(r"doubleClick\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
        "right_click": re.compile(r"rightClick\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
        "type": re.compile(r"(?:write|typewrite)\s*\(\s*['\"](.+?)['\"]\s*\)", re.IGNORECASE),
        "press": re.compile(r"press\s*\(\s*['\"](.+?)['\"]\s*\)", re.IGNORECASE),
        "hotkey": re.compile(r"hotkey\s*\((.+?)\)", re.IGNORECASE),
        "scroll": re.compile(r"scroll\s*\(\s*(-?\d+)\s*(?:,\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+))?\s*\)", re.IGNORECASE),
    }
    
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    
    @classmethod
    def parse(cls, response: str, mode: ExecutionMode = ExecutionMode.HYBRID) -> List[Action]:
        """
        Parse LLM response into a list of actions.
        """
        actions = []
        
        # Check for code blocks first
        code_blocks = cls.CODE_BLOCK_PATTERN.findall(response)
        for lang, code in code_blocks:
            lang = lang.lower() if lang else ""
            code = code.strip()
            
            if not code:
                continue
            
            if lang in ["python", "py"]:
                if "pyautogui" in code.lower():
                    actions.extend(cls._parse_pyautogui_code(code))
                else:
                    actions.append(Action(
                        action_type=ActionType.PYTHON,
                        parameters={"code": code},
                        raw_code=code,
                    ))
            elif lang in ["bash", "sh", "shell"]:
                actions.append(Action(
                    action_type=ActionType.BASH,
                    parameters={"script": code},
                    raw_code=code,
                ))
            else:
                # Try to infer type
                if "pyautogui" in code.lower():
                    actions.extend(cls._parse_pyautogui_code(code))
                else:
                    # Default to bash
                    actions.append(Action(
                        action_type=ActionType.BASH,
                        parameters={"script": code},
                        raw_code=code,
                    ))
        
        # Check for DONE or FAIL
        response_upper = response.upper()
        if "DONE" in response_upper and not any(a.action_type == ActionType.DONE for a in actions):
            actions.append(Action(action_type=ActionType.DONE))
        if "FAIL" in response_upper and not any(a.action_type == ActionType.FAIL for a in actions):
            reason = cls._extract_fail_reason(response)
            actions.append(Action(
                action_type=ActionType.FAIL,
                parameters={"reason": reason},
            ))
        if "INFEASIBLE" in response_upper:
            actions.append(Action(
                action_type=ActionType.FAIL,
                parameters={"reason": "Task is infeasible"},
            ))
        
        return actions
    
    @classmethod
    def _parse_pyautogui_code(cls, code: str) -> List[Action]:
        """Parse PyAutoGUI code into actions."""
        actions = []
        
        for line in code.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("import"):
                continue
            
            for action_name, pattern in cls.PYAUTOGUI_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    if action_name == "click":
                        actions.append(Action(
                            action_type=ActionType.CLICK,
                            parameters={"x": int(match.group(1)), "y": int(match.group(2))},
                            raw_code=line,
                        ))
                    elif action_name == "double_click":
                        actions.append(Action(
                            action_type=ActionType.DOUBLE_CLICK,
                            parameters={"x": int(match.group(1)), "y": int(match.group(2))},
                            raw_code=line,
                        ))
                    elif action_name == "right_click":
                        actions.append(Action(
                            action_type=ActionType.RIGHT_CLICK,
                            parameters={"x": int(match.group(1)), "y": int(match.group(2))},
                            raw_code=line,
                        ))
                    elif action_name == "type":
                        actions.append(Action(
                            action_type=ActionType.TYPE,
                            parameters={"text": match.group(1)},
                            raw_code=line,
                        ))
                    elif action_name == "press":
                        actions.append(Action(
                            action_type=ActionType.PRESS,
                            parameters={"key": match.group(1)},
                            raw_code=line,
                        ))
                    elif action_name == "hotkey":
                        keys = [k.strip().strip("'\"") for k in match.group(1).split(",")]
                        actions.append(Action(
                            action_type=ActionType.HOTKEY,
                            parameters={"keys": keys},
                            raw_code=line,
                        ))
                    elif action_name == "scroll":
                        params = {"amount": int(match.group(1))}
                        if match.group(2) and match.group(3):
                            params["x"] = int(match.group(2))
                            params["y"] = int(match.group(3))
                        actions.append(Action(
                            action_type=ActionType.SCROLL,
                            parameters=params,
                            raw_code=line,
                        ))
                    break
        
        return actions
    
    @classmethod
    def _extract_fail_reason(cls, response: str) -> str:
        """Extract failure reason from response."""
        fail_match = re.search(r"FAIL[:\s]+(.+?)(?:\n|$)", response, re.IGNORECASE)
        if fail_match:
            return fail_match.group(1).strip()
        return "Task failed"


# Utility functions for backward compatibility
def parse_function_args(content: str) -> Dict:
    """Parse function arguments from string."""
    content = content.strip()
    if content.startswith("(") and content.endswith(")"):
        content = content[1:-1]
    
    try:
        return ast.literal_eval("{" + content + "}")
    except:
        args = {}
        for part in content.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    args[key] = ast.literal_eval(value)
                except:
                    args[key] = value
        return args


def extract_first_agent_function(response: str) -> Optional[Tuple[str, Dict]]:
    """Extract the first agent function from LLM response."""
    pattern = r"(\w+)\s*\(([^)]*)\)"
    match = re.search(pattern, response)
    
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        args = parse_function_args(args_str)
        return func_name, args
    
    return None


def parse_single_code_from_string(response: str) -> str:
    """Extract code from a string, handling code blocks."""
    code_block_match = re.search(r"```(?:\w*)\n(.*?)```", response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    inline_match = re.search(r"`([^`]+)`", response)
    if inline_match:
        return inline_match.group(1).strip()
    
    return response.strip()


# Legacy ActionExecutor for backward compatibility
class ActionExecutor:
    """Legacy ActionExecutor class for backward compatibility."""
    
    def __init__(
        self,
        platform: str = "linux",
        width: int = 1920,
        height: int = 1080,
        env_controller = None,
    ):
        self.platform = platform
        self.width = width
        self.height = height
        self.env_controller = env_controller
        
        # 坐标存储（用于 Grounding）
        self.coords1: Optional[Tuple[float, float]] = None
        self.coords2: Optional[Tuple[float, float]] = None
        
        self.executor = HybridExecutor(
            platform=platform,
            width=width,
            height=height,
            env_controller=env_controller,
        )
        self.parser = ActionParser()
    
    def resize_coordinates(self, coords: Optional[Tuple[float, float]]) -> Tuple[int, int]:
        """将归一化坐标转换为像素坐标"""
        if coords is None:
            return (self.width // 2, self.height // 2)
        x, y = coords
        if 0 <= x <= 1 and 0 <= y <= 1:
            return (int(x * self.width), int(y * self.height))
        return (int(x), int(y))
    
    def click(self, instruction: str, num_clicks: int = 1, button_type: str = "left", hold_keys: List[str] = []) -> str:
        """生成点击命令"""
        x, y = self.resize_coordinates(self.coords1)
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        return command
    
    def double_click(self, instruction: str, hold_keys: List[str] = []) -> str:
        """生成双击命令"""
        return self.click(instruction, num_clicks=2, hold_keys=hold_keys)
    
    def right_click(self, instruction: str, hold_keys: List[str] = []) -> str:
        """生成右键点击命令"""
        return self.click(instruction, button_type="right", hold_keys=hold_keys)
    
    def type(self, instruction: str, text: str, overwrite: bool = False, enter: bool = False) -> str:
        """
        生成输入文本命令

        Args:
            instruction: 目标元素描述（目前只用来配合 coords1 点击）
            text: 要输入的文本
            overwrite: 是否覆盖现有内容（先全选再输入）
            hold_keys: 按住的键
            enter: 是否在输入完成后按一次回车
        """
        x, y = self.resize_coordinates(self.coords1)

        command = "import pyautogui; import time; "

        # 先点击定位

        command += f"pyautogui.click({x}, {y}); "
        command += "time.sleep(0.1); "

        # 如果需要覆盖，先全选现有内容
        if overwrite:
            command += "pyautogui.hotkey('ctrl', 'a'); "
            command += "time.sleep(0.1); "

        # 输入文本
        if text:
            command += f"pyautogui.typewrite({repr(text)}, interval=0.05); "

        # 根据 enter 标志按回车
        if enter:
            command += "time.sleep(0.5); pyautogui.press('enter'); "

        return command
    
    def scroll(self, instruction: str, amount: int = -3) -> str:
        """生成滚动命令"""
        x, y = self.resize_coordinates(self.coords1)
        return f"import pyautogui; pyautogui.scroll({amount}, x={x}, y={y})"
    
    def drag_and_drop(self, start_instruction: str, end_instruction: str, hold_keys: List[str] = []) -> str:
        """生成拖拽命令"""
        x1, y1 = self.resize_coordinates(self.coords1)
        x2, y2 = self.resize_coordinates(self.coords2)
        
        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.drag({x2 - x1}, {y2 - y1}, duration=0.5); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        return command
    
    def hotkey(self, keys) -> str:
        """
        生成快捷键命令
        
        Args:
            keys: 按键列表，如 ['ctrl', 'a']，或单个按键字符串
        """
        # 确保 keys 是列表
        if isinstance(keys, str):
            # 如果是字符串表示的列表，尝试解析
            if keys.startswith('[') and keys.endswith(']'):
                import ast
                try:
                    keys = ast.literal_eval(keys)
                except:
                    keys = [keys]
            else:
                keys = [keys]
        
        # 将列表转换为多个独立的参数
        keys_str = ", ".join(repr(k) for k in keys)
        return f"import pyautogui; pyautogui.hotkey({keys_str})"
    
    def wait(self, seconds: float = 1.0) -> str:
        """生成等待命令"""
        return f"import time; time.sleep({seconds})"
    
    def code(self, python_code: str) -> str:
        """
        直接执行 Python 代码
        
        Args:
            python_code: 要执行的 Python 代码字符串
            
        Returns:
            执行的代码（原样返回，由上层执行）
        """
        # 清理代码中的多余引号和换行
        code = python_code.strip()
        # 标记这是直接执行的代码
        return f"__EXEC_CODE__:{code}"
    
    def done(self) -> str:
        """任务完成"""
        return "DONE"
    
    def fail(self, reason: str = "") -> str:
        """任务失败"""
        return f"FAIL:{reason}" if reason else "FAIL"
