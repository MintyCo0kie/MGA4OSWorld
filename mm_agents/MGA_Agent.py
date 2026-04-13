"""
MGA Agent - Memory-Driven GUI Agent
"""

import base64
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import time
import backoff
import numpy as np
import cv2
from PIL import Image
from requests.exceptions import SSLError

from mm_agents.actions import (
    ActionExecutor,
    parse_function_args,
    extract_first_agent_function,
    parse_single_code_from_string,
)
from mm_agents.memory_agent import MemoryAgent
from mm_agents.observer_agent import ObserverAgent
from mm_agents.grounding_agent import GroundingAgent
from mm_agents.prompts import  GTA1_PLANNER_GUI_GROUNDING_SYSTEM_PROMPT
from mm_agents.llm_server import OpenAIClient, LocalModelClient
from mm_agents.config import load_config, MGA_Config
from mm_agents.utils.kimi_utils import parse_response_to_cot_and_action

logger = logging.getLogger("desktopenv.agent.run")


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def encode_numpy_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def split_to_batches(n: int, batch_size: int = 8) -> List[int]:
    batches = [batch_size] * (n // batch_size)
    if n % batch_size:
        batches.append(n % batch_size)
    return batches


def extract_code_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = r"```(\w+)\n([\s\S]*?)```"
    match = re.search(pattern, text)

    if match:
        code_type = match.group(1).lower()
        code_content = match.group(2).strip()
        logger.debug(f"Regex match successful: type={code_type}")
        return code_type, code_content

    logger.warning("No standard markdown code block found via regex.")
    return None, None


class MGA1Agent:

    def __init__(
        self,
        env,
        config_path: Optional[str] = None,
        config: Optional[MGA_Config] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        cfg = self.config

        self.platform = cfg.common.platform
        self.max_tokens = cfg.common.max_tokens
        self.temperature = cfg.common.temperature
        self.top_p = cfg.common.top_p
        self.action_space = cfg.common.action_space
        self.observation_type = cfg.common.observation_type
        self.max_steps = cfg.common.max_steps
        self.max_image_history_length = cfg.common.max_image_history_length
        self.N_SEQ = cfg.common.N_SEQ
        self.width = cfg.common.width
        self.height = cfg.common.height

        self.screen_size = cfg.common.screen_size
        self.coordinate_type = cfg.common.coordinate_type
        self.thinking = cfg.common.thinking
        
        assert self.action_space in ["pyautogui"], "Invalid action space"
        assert self.observation_type in ["screenshot"], "Invalid observation type"

        self.env = env



        self.Planner_prompt = GTA1_PLANNER_GUI_GROUNDING_SYSTEM_PROMPT

        if cfg.planner.client_type == "openai":
            self.planner_client = OpenAIClient(
                model=cfg.planner.model,
                api_key=cfg.planner.api_key,
                base_url=cfg.planner.base_url,
                temperature=self.temperature,
            )
        else:
            self.planner_client = LocalModelClient(
                model=cfg.planner.model,
                base_url=cfg.planner.base_url,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        logger.info(f"Planner initialized: {cfg.planner.client_type}, {cfg.planner.model}")

        self.need_memory = cfg.memory.enabled

        if self.need_memory:
            self.memory: List[str] = []
            self.memory_agent = MemoryAgent(
                client_type=cfg.memory.client_type,
                model=cfg.memory.model,
                base_url=cfg.memory.base_url,
                api_key=cfg.memory.api_key,
                max_tokens=self.max_tokens,
            )
            logger.info(f"Memory initialized: {cfg.memory.client_type}, {cfg.memory.model}")
        else:
            self.memory_agent = None
            logger.info("Memory disabled")

        self.need_observer = cfg.observer.enabled

        if self.need_observer:
            self.observations: List[str] = []
            self.observer_agent = ObserverAgent(
                client_type=cfg.observer.client_type,
                model=cfg.observer.model,
                base_url=cfg.observer.base_url,
                api_key=cfg.observer.api_key,
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
            logger.info(f"Observer initialized: {cfg.observer.client_type}, {cfg.observer.model}")
        else:
            self.observer_agent = None
            logger.info("Observer disabled")

        self.grounding_agent = GroundingAgent(
            client_type=cfg.grounding.client_type,
            model=cfg.grounding.model,
            base_url=cfg.grounding.base_url,
            api_key=cfg.grounding.api_key,
            width=self.width,
            height=self.height,
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        logger.info(f"Grounding initialized: {cfg.grounding.client_type}, {cfg.grounding.model}")

        self.action_executor = ActionExecutor(self.platform, self.width, self.height)

        if self.planner_client.model.startswith("gpt-5.4"):
            self.observation_captions: List[str] = []
        self.screenshots: List[bytes] = []
        self.thoughts: List[str] = []
        self.actions = []

        self.current_step: int = 0

        logger.info("MGA1Agent fully initialized")

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        error = ""
        scripts = ""
        observation = ""
        summary = ""

        raw_screenshot: bytes = obs.get("screenshot")

        if self.need_observer and self.observer_agent is not None:
            observation = self.observer_agent.observe(raw_screenshot)

        if self.need_memory and self.memory_agent is not None:
            if len(self.thoughts) > 0 and len(self.screenshots) > 0:
                last_screenshot = self.screenshots[-1] if len(self.screenshots) >= 1 else None
                summary = self.memory_agent.summarize(
                    instruction=instruction,
                    recent_actions=str(self.actions),
                    recent_reasoning=str(self.thoughts),
                    last_error=error,
                    last_screenshot=last_screenshot,
                    last_memory=self.memory[-1] if getattr(self, "memory", None) else "No previous steps.",
                    current_screenshot=raw_screenshot,
                    current_obs = observation
                ).split("</think>")[-1]
            else:
                summary = "No previous steps. This is the first action."

            if not hasattr(self, "memory"):
                self.memory = []
            self.memory.append(summary)

        user_prompt = self._build_planner_prompt(
            instruction=instruction,
            observation=observation if self.need_observer else None,
            memory=summary if self.need_memory else None,
        )

        if "agent.code" in self.actions[-1] if self.actions else []:
            user_prompt += f"\n\n## Last Execution result(not always execution failed)\n{error}\n"

        if self.need_memory:
            messages = self.planner_client.build_messages(
                system_prompt= self.Planner_prompt,
                user_prompt=user_prompt,
                images=[raw_screenshot],
            )
        else:
            messages: List[Dict[str, Any]] = [
                self.planner_client.build_message("system", self.Planner_prompt)
            ]

            use_caption_history = self.planner_client.model.startswith("gpt-5.4") and hasattr(self, "observation_captions")

            obs_start_idx = max(0, len(self.thoughts) - int(self.max_image_history_length or 0))

            for i in range(len(self.thoughts)):
                if use_caption_history and i < obs_start_idx:
                    caption = ""
                    if i < len(self.observation_captions):
                        caption = self.observation_captions[i] or ""
                    hist_text_parts = [f"Step {i+1}:"]
                    if caption:
                        hist_text_parts.append(f"Observation: {caption}")
                    hist_text = "\n".join(hist_text_parts).strip() + "\n"
                    messages.append(
                        self.planner_client.build_message(
                            "user",
                            hist_text,
                            images=None,
                        )
                    )
                else:
                    hist_text_parts = [f"Step {i+1}:"]
                    if self.need_observer and i < len(getattr(self, "observations", [])):
                        hist_text_parts.append(f"Observation:\n{self.observations[i]}")
                    hist_text_parts.append("Screenshot:")
                    hist_text = "\n".join(hist_text_parts) + "\n"

                    if i < len(self.screenshots):
                        messages.append(
                            self.planner_client.build_message(
                                "user",
                                hist_text,
                                images=[self.screenshots[i]],
                            )
                        )
                    else:
                        messages.append(
                            self.planner_client.build_message(
                                "user",
                                "\n".join(hist_text_parts[:-1]).strip() + "\n",
                                images=None,
                            )
                        )

                thought_messages = f"Thought:\n{self.thoughts[i]}"
                action_messages = (
                    "Action:\n" + "\n".join(map(str, self.actions[i]))
                    if i < len(self.actions)
                    else "Action:\n"
                )
                messages.append(
                    self.planner_client.build_message(
                        "assistant",
                        thought_messages + "\n\n" + action_messages,
                    )
                )

            messages.append(
                self.planner_client.build_message(
                    "user",
                    user_prompt,
                    images=[raw_screenshot],
                )
            )

        codes, pyautogui_actions, thought, scripts, error = self._generate_planner_response(messages, obs)

        self.screenshots.append(raw_screenshot)
        if self.need_observer:
            if not hasattr(self, "observations"):
                self.observations = []
            self.observations.append(observation)

        if self.planner_client.model.startswith("gpt-5.4"):
            self.observation_captions.append(observation_caption)

        self.thoughts.append(thought)
        self.actions.append(codes)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            pyautogui_actions = ["FAIL"]

        logger.debug("=" * 80)
        if self.need_observer and observation:
            logger.debug("-" * 40)
            logger.debug(f"Step {self.current_step}: Observation - {observation}")
        if self.need_memory and summary:
            logger.debug("-" * 40)
            logger.debug(f"Step {self.current_step}: Memory - {summary}")
        logger.debug("-" * 40)
        logger.debug(f"Step {self.current_step}: Thought - {thought}")
        logger.debug("-" * 40)
        logger.debug(f"Step {self.current_step}: Actions - {codes}")
        if codes and "agent.code" in codes[0]:
            logger.debug("-" * 40)
            logger.debug(f"Step {self.current_step}: Scripts - {scripts}")
            logger.debug("-" * 40)
            logger.debug(f"Step {self.current_step}: Execution Error - {error}")
        logger.debug("=" * 80)

        return thought, pyautogui_actions

    def _get_current_observation_description(self, obs: Dict) -> str:
        if not self.need_observer:
            return ""
        try:
            description = self.observer_agent.observe(obs)
            if description:
                return description
        except Exception as e:
            logger.debug(f"Observer error: {e}")
        return ""
    

    def _execute_code(self, scripts: str, code_type: str = "python") -> str:
        if not scripts:
            return "No scripts to execute"

        logger.info(f"--- [EXECUTOR] Logic Start | Type: {code_type} ---")
        logger.debug(f"Scripts to execute:\n{scripts}")

        if code_type in ["python", "python3", "py"]:
            raw_res = self.env.controller.run_python_script(scripts)
        elif code_type in ["bash", "sh", "shell"]:
            raw_res = self.env.controller.run_bash_script(scripts)
        else:
            logger.warning(f"Unsupported language type: {code_type}")
            return f"Unsupported language type: {code_type}"

        logger.info(f"Raw execution result: {raw_res}")
        logger.info(f"--- [EXECUTOR] Logic End | Type: {code_type} ---")

        if raw_res.get("status", "") == "success":
            return raw_res.get("message", "")
        else:
            return raw_res.get("output", "Execution error")

    def _build_planner_prompt(self, instruction: str=None, observation: str=None, memory: str=None) -> str:
        planner_prompt = f"""
        First you should refer to the previous actions and observations for reflection, then generate the next move according to the UI screenshot and instruction. \n\nInstruction: {instruction}\n\n
"""
        
        if observation:
            planner_prompt += f"""
## Current Screen Observation 
{observation}
"""
        if memory != "":
            planner_prompt += f"""
## Memory Context 
{memory}
"""
        planner_prompt += f"""
#     - **Password**: Your sudo password is "password".
#     - **User**: Your username is "user".
#     - **Home**: Your home path is "/home/user".
# """


        return planner_prompt

    def _generate_planner_response(self, messages: List[Dict], obs) -> Tuple:
        max_retry = 5
        retry_count = 0
        error = ""
        scripts = ""

        while retry_count < max_retry:
            try:
                response = self.planner_client.call(messages)

                if not self._is_valid_response(response):
                    breakpoint()
                    retry_count += 1
                    logger.warning(f"Invalid response, retrying ({retry_count}/{max_retry})...")
                    continue

                codes = self._parse_code_from_response(response)
                thought = self._parse_thought_from_response(response)
                scripts, code_type = self._parse_scripts_from_response(response)

                is_direct_code = (
                    scripts is not None
                    or (codes and re.search(r'agent\.code\s*\(', codes[0]))
                )

                if is_direct_code:
                    if scripts:
                        error = self._execute_code(scripts, code_type)
                    else:
                        error = "Scripts block missing for agent.code() action"
                        logger.warning(error)
                    pyautogui_actions = [""]
                else:
                    error = ""
                    scripts = ""
                    pyautogui_actions = self._ground_and_execute(response, codes, obs)

                return codes, pyautogui_actions, thought, scripts or "", error

            except Exception as e:
                retry_count += 1
                logger.error(f"Planner response error: {e}, retrying ({retry_count}/{max_retry})...")

        logger.error("Max retries exceeded, returning FAIL")
        return ["agent.fail()"], ["FAIL"], "", "", "Max retries exceeded"

    def _ground_and_execute(self, planner_response: str, codes: List[str], obs: Dict) -> List[str]:
        if not codes:
            logger.warning("No codes to execute")
            return ["FAIL"]
        
        code_str = codes[0] if isinstance(codes, list) else codes
        
        if not code_str:
            return ["FAIL"]
        
        agent_call = self._extract_agent_call(code_str)
        if not agent_call:
            logger.warning(f"No agent call found in: {code_str[:100]}")
            return ["FAIL"]
        
        function_name, args = agent_call
        logger.debug(f"Parsed action: {function_name}, args: {[str(a)[:] for a in args]}")
        
        if function_name == "done":
            return ["DONE"]
        
        elif function_name == "fail":
            reason = args[0] if args else ""
            logger.info(f"Task failed: {reason}")
            return ["FAIL"]
        
        elif function_name == "wait":
            seconds = float(args[0]) if args else 1.0
            return [f"import time; time.sleep({seconds})"]
        
        elif function_name == "hotkey":
            return self._generate_hotkey_code(args)
        
        elif function_name in ["click", "double_click", "right_click", "type", "scroll", "drag_and_drop"]:
            return self._generate_gui_action_code(function_name, args, obs)
        
        else:
            logger.warning(f"Unknown action: {function_name}")
            return ["FAIL"]
    
    def _generate_hotkey_code(self, args: List[str]) -> List[str]:
        import ast
        
        if not args:
            return ["FAIL"]
        
        keys = args[0] if len(args) == 1 else args
        
        if isinstance(keys, str):
            if keys.startswith('[') and keys.endswith(']'):
                try:
                    keys = ast.literal_eval(keys)
                except (ValueError, SyntaxError):
                    keys = [keys]
            else:
                keys = [keys]
        
        if not isinstance(keys, list):
            keys = [keys]
        
        keys_repr = ", ".join(repr(k) for k in keys)
        code = f"import pyautogui; pyautogui.hotkey({keys_repr})"
        logger.info(f"Hotkey code: {code}")
        return [code]
    
    def _generate_gui_action_code(self, function_name: str, args: List[str], obs: Dict) -> List[str]:
        self.action_executor.coords1 = None
        self.action_executor.coords2 = None
        
        if function_name in ["click", "double_click", "right_click"]:
            if args:
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
                    logger.debug(f"Grounded '{args[0][:30]}...' -> {coords}")
                else:
                    logger.warning(f"Grounding failed for: {args[0][:]}")
                    return ["FAIL"]
        
        elif function_name == "type":
            if args:
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
                    logger.debug(f"Grounded '{args[0][:30]}...' -> {coords}")
        
        elif function_name == "scroll":
            if args and not args[0].lstrip('-').isdigit():
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
        
        elif function_name == "drag_and_drop":
            if len(args) >= 2:
                coords1 = self.grounding_agent.ground_normalized(args[0], obs)
                coords2 = self.grounding_agent.ground_normalized(args[1], obs)
                
                if coords1:
                    self.action_executor.coords1 = coords1
                    logger.debug(f"Grounded start '{args[0][:30]}...' -> {coords1}")
                else:
                    logger.warning(f"Grounding failed for start: {args[0][:]}")
                    return ["FAIL"]
                
                if coords2:
                    self.action_executor.coords2 = coords2
                    logger.debug(f"Grounded end '{args[1][:30]}...' -> {coords2}")
                else:
                    logger.warning(f"Grounding failed for end: {args[1][:]}")
                    return ["FAIL"]
        
        try:
            executor = self.action_executor
            
            if function_name == "click":
                num_clicks = 1
                button_type = 'left'
                if len(args) > 1:
                    for arg in args[1:]:
                        if isinstance(arg, str) and (arg.startswith('num_clicks=') or arg.startswith('clicks=')):
                            try:
                                num_clicks = int(arg.split('=')[1])
                            except:
                                pass
                        elif isinstance(arg, str) and (arg.startswith('button_type=') or arg.startswith('button=') or arg.startswith('btn=')):
                            button_type = arg.split('=')[1].strip("'\"")
                command = executor.click(args[0] if args else "", num_clicks=num_clicks, button_type=button_type)
            elif function_name == "double_click":
                command = executor.double_click(args[0] if args else "")
            elif function_name == "right_click":
                command = executor.right_click(args[0] if args else "")
            elif function_name == "type":
                target = args[0] if args else ""
                text = ""
                overwrite = False
                enter = False

                if len(args) > 1:
                    for arg in args[1:]:
                        if isinstance(arg, str) and arg.startswith('text='):
                            text_part = arg.split('text=', 1)[1]
                            if (text_part.startswith('"') and text_part.endswith('"')) or \
                               (text_part.startswith("'") and text_part.endswith("'")):
                                text = text_part[1:-1]
                            else:
                                text = text_part
                        elif arg.startswith('overwrite='):
                            overwrite = arg.split('overwrite=')[1].strip("'\"") in ["True","true", "1", "yes"]
                        elif arg.startswith('enter='):
                            enter = arg.split('enter=')[1].strip("'\"") in ["True","true", "1", "yes"]


                command = executor.type(target, text, overwrite=overwrite, enter=enter)

            elif function_name == "scroll":
                amount = int(args[1]) if len(args) > 1 and args[1].lstrip('-').isdigit() else -3
                if args and args[0].lstrip('-').isdigit():
                    amount = int(args[0])
                command = executor.scroll("", amount)
            elif function_name == "drag_and_drop":
                command = executor.drag_and_drop(
                    args[0] if args else "",
                    args[1] if len(args) > 1 else ""
                )
            else:
                return ["FAIL"]
            
            logger.debug(f"Generated command: {command}")
            return [command]
            
        except Exception as e:
            logger.error(f"GUI action generation error: {e}")
            return ["FAIL"]
    
    def _extract_agent_call(self, code_str: str) -> Optional[Tuple[str, List[str]]]:
        match = re.search(r'agent\.(\w+)\s*\((.*)\)', code_str, re.DOTALL)
        if not match:
            return None
        
        function_name = match.group(1)
        args_str = match.group(2).strip()
        
        args = self._parse_function_args(args_str)
        
        return (function_name, args)


    def _extract_pyautogui_call(self, code_str: str) -> Optional[Tuple[str, List[str]]]:
        match = re.search(r'pyautogui\.(\w+)\s*\((.*)\)', code_str, re.DOTALL)
        if not match:
            return None
        
        function_name = match.group(1)
        args_str = match.group(2).strip()
        
        args = self._parse_function_args(args_str)
        
        return (function_name, args)

    
    def _parse_function_args(self, args_str: str) -> List[str]:
        if not args_str:
            return []
        
        args = []
        current_arg = ""
        depth = 0
        in_string = False
        string_char = None
        i = 0
        
        while i < len(args_str):
            char = args_str[i]
            
            if char == '\\' and i + 1 < len(args_str):
                current_arg += char + args_str[i + 1]
                i += 2
                continue
            
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current_arg += char
            
            elif char in ['(', '[', '{'] and not in_string:
                depth += 1
                current_arg += char
            elif char in [')', ']', '}'] and not in_string:
                depth -= 1
                current_arg += char
            
            elif char == ',' and depth == 0 and not in_string:
                arg = self._clean_arg(current_arg)
                if arg is not None:
                    args.append(arg)
                current_arg = ""
            
            else:
                current_arg += char
            
            i += 1
        
        arg = self._clean_arg(current_arg)
        if arg is not None:
            args.append(arg)
        
        return args
    
    def _clean_arg(self, arg: str) -> Optional[str]:
        arg = arg.strip()
        if not arg:
            return None
        
        if (arg.startswith('"') and arg.endswith('"')) or \
           (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]
        
        if arg == "[]":
            return None
        
        return arg
    
    def _execute_gui_action(self, function_name: str, args: List[str], obs: Dict) -> List[str]:
        self.action_executor.coords1 = None
        self.action_executor.coords2 = None
        
        if function_name in ["click", "double_click", "right_click"]:
            if args:
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
                    logger.debug(f"Grounded '{args[0][:30]}...' -> {coords}")
                else:
                    logger.warning(f"Grounding failed for: {args[0][:]}")
                    return ["FAIL"]
        
        elif function_name == "type":
            if args:
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
                    logger.debug(f"Grounded '{args[0][:30]}...' -> {coords}")
        
        elif function_name == "scroll":
            if args and not args[0].lstrip('-').isdigit():
                coords = self.grounding_agent.ground_normalized(args[0], obs)
                if coords:
                    self.action_executor.coords1 = coords
        
        elif function_name == "drag_and_drop":
            if len(args) >= 2:
                coords1 = self.grounding_agent.ground_normalized(args[0], obs)
                coords2 = self.grounding_agent.ground_normalized(args[1], obs)
                
                if coords1:
                    self.action_executor.coords1 = coords1
                    logger.debug(f"Grounded start '{args[0][:30]}...' -> {coords1}")
                else:
                    logger.warning(f"Grounding failed for start: {args[0][:]}")
                    return ["FAIL"]
                
                if coords2:
                    self.action_executor.coords2 = coords2
                    logger.debug(f"Grounded end '{args[1][:30]}...' -> {coords2}")
                else:
                    logger.warning(f"Grounding failed for end: {args[1][:]}")
                    return ["FAIL"]
        
        try:
            executor = self.action_executor
            
            if function_name == "click":
                num_clicks = 1
                button_type = 'left'
                if len(args) > 1:
                    for arg in args[1:]:
                        if isinstance(arg, str) and arg.startswith('num_clicks='):
                            try:
                                num_clicks = int(arg.split('=')[1])
                            except:
                                pass
                        elif isinstance(arg, str) and arg.startswith('button_type='):
                            button_type = arg.split('=')[1].strip("'\"")
                command = executor.click(args[0] if args else "", num_clicks=num_clicks, button_type=button_type)
            elif function_name == "double_click":
                command = executor.double_click(args[0] if args else "")
            elif function_name == "right_click":
                command = executor.right_click(args[0] if args else "")
            elif function_name == "type":
                target = args[0] if args else ""
                text = args[1] if len(args) > 1 else ""
                overwrite = False
                
                if text and ((text.startswith('"') and text.endswith('"')) or 
                             (text.startswith("'") and text.endswith("'"))):
                    text = text[1:-1]  
            
                if len(args) > 2:
                    overwrite_arg = args[2].lower().strip()
                    overwrite = overwrite_arg in ["true", "1", "yes"]
                
                command = executor.type(target, text, overwrite=overwrite)
            elif function_name == "scroll":
                amount = int(args[1]) if len(args) > 1 and args[1].lstrip('-').isdigit() else -3
                if args and args[0].lstrip('-').isdigit():
                    amount = int(args[0])
                command = executor.scroll("", amount)
            elif function_name == "drag_and_drop":
                command = executor.drag_and_drop(
                    args[0] if args else "",
                    args[1] if len(args) > 1 else ""
                )
            else:
                return ["FAIL"]
            
            logger.debug(f"Generated command: {command[:100]}")
            return self._execute_pyautogui_code(command)
            
        except Exception as e:
            logger.error(f"GUI action error: {e}")
            return ["FAIL"]
    
    
    def _parse_code_from_response(self, input_string: str) -> List[str]:
        action_pattern = r"###\s*Action:\s*```(?:python|code)?\s*(.*?)```"
        matches = re.findall(action_pattern, input_string, re.DOTALL)

        if not matches:
            action_plain_pattern = r"(?:###\s*)?Action:\s*\n(.*?)(?=###\s*\w+:|^Action:|Scripts|\Z)"
            plain_matches = re.findall(action_plain_pattern, input_string, re.DOTALL)
            if plain_matches:
                for block in plain_matches:
                    calls = self._extract_all_agent_calls(block)
                    matches.extend(calls)

        if not matches:
            fallback_pattern = r"```(?:python|code)?\s*(.*?)```"
            matches = re.findall(fallback_pattern, input_string, re.DOTALL)

        if not matches:
            calls = self._extract_all_agent_calls(input_string)
            matches.extend(calls)

        seen = set()
        result = []
        for match in matches:
            code = match.strip()
            if code and code not in seen:
                seen.add(code)
                result.append(code)
        return result


        if not matches:
            fallback_pattern = r"```(?:python|code)?\s*(.*?)```"
            matches = re.findall(fallback_pattern, input_string, re.DOTALL)

        seen = set()
        result = []
        for match in matches:
            code = match.strip()
            if code and code not in seen:
                seen.add(code)
                result.append(code)
        return result


    def _extract_all_agent_calls(self, text: str) -> List[str]:
        results = []
        for m in re.finditer(r'agent\.\w+\s*\(', text):
            start = m.start()
            paren_start = m.end() - 1  
            
            depth = 0
            in_string = False
            string_char = None
            i = paren_start
            
            while i < len(text):
                char = text[i]
                
                if char == '\\' and in_string and i + 1 < len(text):
                    i += 2
                    continue
                
                if char in ('"', "'") and not in_string:
                    in_string = True
                    string_char = char
                elif in_string and char == string_char:
                    in_string = False
                    string_char = None
                
                elif not in_string:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            results.append(text[start:i + 1])
                            break
                i += 1
        
        return results

    def _parse_scripts_from_response(self, input_string: str) -> Tuple[Optional[str], str]:
        scripts_pattern = r"###\s*Scripts.*?:\s*```(scripts|python|python3|sh|bash)?\s*(.*?)```"
        match = re.search(scripts_pattern, input_string, re.DOTALL)

        if match:
            code_type = (match.group(1) or "python").strip().lower()
            if code_type in ["scripts", "python3"]:
                code_type = "python"
            scripts = match.group(2).strip()
            return (scripts if scripts else None), code_type

        return None, "python"

    def _parse_thought_from_response(self, input_string: str) -> str:
        pattern = r"###\s*Thought:\s*(.*?)(?=###\s*Action:|$)"
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        fallback_pattern = r"Thought:(.*?)\n(?:Action:|Step|```)"
        matches = re.findall(fallback_pattern, input_string, re.DOTALL)
        return matches[0].strip() if matches else ""
    
    def parse_observation_caption_from_planner_response(self, input_string: str) -> str:
        pattern = r"Observation:\n(.*?)\n"
        matches = re.findall(pattern, input_string, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    def reset(self, _logger=None) -> None:
        global logger
        if _logger:
            logger = _logger
        else:
            logger = logging.getLogger("desktopenv.agent.run")
        
        self.current_step = 0
        self.thoughts = []
        self.actions = []
        self.screenshots = []
        if self.need_observer:
            self.observations = []
        if self.need_memory:
            self.memory = []
            self.memory_agent.reset()
        logger.debug("MGA1Agent reset")
    
    def _is_valid_response(self, response: str) -> bool:
        codes = self._parse_code_from_response(response)
        thought = self._parse_thought_from_response(response)
        return bool(codes and thought)
    
    def _check_repeated_actions(self) -> Optional[str]:
        if len(self.actions) < 3:
            return None
        
        recent_actions = self.actions[-3:]
        
        type_pattern = r'agent\.type\(["\']([^"\']+)["\']'
        targets = []
        for action in recent_actions:
            match = re.search(type_pattern, str(action))
            if match:
                targets.append(match.group(1)[:30])  
        
        if len(targets) == 3 and len(set(targets)) == 1:
            return f"You have typed into '{targets[0]}' 3 times in a row. The input may not be working correctly. Try a DIFFERENT approach: use hotkey (Ctrl+A to select all, then type), or use code method with specific coordinates, or try a different element."
        
        action_strs = [str(a)[:100] for a in recent_actions]
        if len(set(action_strs)) == 1:
            return f"You have repeated the exact same action 3 times. This is not working. Try a COMPLETELY DIFFERENT approach or call agent.fail() if the task cannot be completed."
        
        return None
    
    def _parse_action(self, response: str) -> Tuple[str, List[Any]]:
        import ast
        
        pattern = r'agent\.(\w+)\((.*?)\)(?:\s*$|\s*```)'
        match = re.search(pattern, response, re.DOTALL)
        
        if not match:
            return "none", []
        
        function_name = match.group(1)
        args_str = match.group(2).strip()
        
        if not args_str:
            return function_name, []
        
        try:
            parsed_args = ast.literal_eval(f"{{{args_str}}}")
            args = [parsed_args]
            
            logger.debug(f"Parsed action: {function_name}, args: {args}")
            return function_name, args
            
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse args with ast: {e}, falling back to string")
            args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
            return function_name, args
    
    def _generate_action_code(self, function_name: str, args: List[Any], obs: Dict) -> List[str]:
        if function_name == "hotkey":
            keys = args[0] if args else []
            
            if isinstance(keys, str):
                import ast
                try:
                    keys = ast.literal_eval(keys)
                except:
                    keys = [keys]
            
            command = self.action_executor.hotkey(keys)
            logger.info(f"Hotkey code: {command}")
            return [command]
        
from openai import OpenAI

def add_box_token(input_string):
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  
            for coord_type, x, y in coordinates:
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


