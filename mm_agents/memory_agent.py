"""
Memory Agent - 执行历史总结和上下文维护
"""

import logging
import os
from typing import Dict, List, Literal, Optional

import backoff

from mm_agents.llm_server import OpenAIClient, LocalModelClient, LLMClient
from mm_agents.prompts import SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger("desktopenv.agent.memory")




class MemoryAgent:


    def __init__(
        self,
        client_type: Literal["local", "openai"] = "local",
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.client_type = client_type
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.client: LLMClient = self._create_client()
        self.memory_history: List[str] = []
        self.action_history: List[str] = []  
        logger.info(f"MemoryAgent initialized: {client_type}, {model}")

    def _create_client(self) -> LLMClient:
        if self.client_type == "openai":
            return OpenAIClient(
                model=self.model, api_key=self.api_key, base_url=self.base_url,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
        return LocalModelClient(
            model=self.model, base_url=self.base_url,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )

    def summarize(
        self,
        instruction: str,
        recent_reasoning: str,
        recent_actions: str,
        last_error: str,
        last_screenshot: str,
        last_memory: str,
        current_screenshot: str,
        current_obs:str,
        current_step: int = 0,
        max_steps: int = 30,
    ) -> str:
        """
        生成当前步骤的记忆摘要（仅记录事实，不给建议）
        """
        
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return [str(i) for i in x]
            return [str(x)]

        recent_actions_list = _to_list(recent_actions)
        recent_reasoning_list = _to_list(recent_reasoning)

        last_thought = recent_reasoning_list[-1] if recent_reasoning_list else "N/A"
        last_action = recent_actions_list[-1] if recent_actions_list else "N/A"


        user_prompt = f"""
- **Instruction (The overarching goal the agent is trying to achieve) : is: (`{instruction}`)**

- **Last step  (The most recent command and its feedback)** is: 
          `Last action`:(`{last_action}`)
          `Last thought`:(`{last_thought}`)

- **Historical Memory(The baseline state summary of previous tasks) is: (`{last_memory}`)
- **Screenshot Context**:
  - **Image 1**: Previous State (Before the most recent action).
  - **Image 2**: Current State (After the most recent action).
  - If only one image is provided, it represents the current state, and no previous state for comparison, thus None Issue.



---
"""
        if current_obs != "":
            user_prompt += f"""
## current screenshot visual information observation:
    {current_obs}

"""

        if last_error != "":

            user_prompt += f"""
## Last Execution for code feedback

{last_error}

If the last action was a code execution and information show that code execution successfully, you should show that code execution is successful must show that mustn't repeat code execution for same idea.
"""
        if last_screenshot == None:
            messages = self.client.build_messages(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=[current_screenshot],
            )
        else:
            messages = self.client.build_messages(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=[last_screenshot, current_screenshot],
            )
        
        summary = self._call_llm(messages)
        self.memory_history.append(summary)
        return summary

    @backoff.on_exception(backoff.constant, Exception, interval=5, max_tries=3)
    def _call_llm(self, messages: List[Dict]) -> str:
        responses = self.client.call(messages)
        if self.model.startswith("gpt"):
            return responses if responses else ""
        return responses[0] if responses else ""
    
    def get_latest_memory(self) -> str:
        return self.memory_history[-1] if self.memory_history else "No previous steps."
    
    def get_full_history(self) -> List[str]:
        return self.memory_history.copy()
    
    def reset(self) -> None:
        self.memory_history = []
        self.action_history = []  
