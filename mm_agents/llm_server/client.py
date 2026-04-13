"""
LLM Client implementations for different API providers.
"""

import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import backoff
import requests
import cv2
from PIL import Image
from requests.exceptions import SSLError

logger = logging.getLogger("desktopenv.llm_server.client")


class LLMClient(ABC):
    def __init__(self, model: str, max_tokens: int = 4096, temperature: float = 0.1):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def call(self, messages: List[Dict], n: int = 1, **kwargs) -> List[str]:
        pass

    def _encode_image(self, image: Any) -> str:
        """统一图片编码：接受 bytes 或 str(base64)，返回 base64 str"""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        elif isinstance(image, str):
            # 已经是 base64 字符串，直接返回
            return image
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise ValueError("Failed to encode numpy image")
            return base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def build_message(
        self,
        role: str,
        text: str,
        images: Optional[List[Any]] = None,
    ) -> Dict:
        """构建单条消息，子类可覆盖"""
        if images:
            content = [{"type": "text", "text": text}]
            for image in images:
                encoded = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                        "detail": "high"
                    }
                })
            return {"role": role, "content": content}
        return {"role": role, "content": text}

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[Any]] = None,
    ) -> List[Dict]:
        """构建完整消息列表：system + user（带图片）"""
        messages = [self.build_message("system", system_prompt)]
        messages.append(self.build_message("user", user_prompt, images))
        return messages


class OpenAIClient(LLMClient):

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        super().__init__(model, max_tokens, temperature)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    # build_message 和 build_messages 继承自 LLMClient，格式完全兼容 OpenAI API

    @backoff.on_exception(
        backoff.constant,
        (SSLError, requests.exceptions.RequestException),
        interval=30,
        max_tries=10,
    )
    def call(self, messages: List[Dict], n: int = 1, **kwargs) -> List[str]:
        if not self.is_available():
            logger.error("OpenAI API key or base URL not configured")
            return [""]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
            },
        )
        response = response.json()  
        try:
            if self.model.startswith("kimi") :
                finish_reason = response["choices"][0].get("finish_reason")
                if finish_reason is not None and finish_reason == "stop": # for most of the time, length will not exceed max_tokens
                    return response['choices'][0]['message']

            return [choice["message"]["content"] for choice in response["choices"]][0]
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""


class LocalModelClient(LLMClient):
    def __init__(
        self,
        model: str = "qwen",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.9,
        timeout: int = 300,
    ):
        super().__init__(model, max_tokens, temperature)
        self.base_url = base_url or os.environ.get("LOCAL_MODEL_URL", "")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.base_url)

    def build_message(
        self,
        role: str,
        text: str,
        images: Optional[List[Any]] = None,
    ) -> Dict:
        """LocalModel 使用 image/image_base64 格式"""
        if images:
            content = [{"type": "text", "text": text}]
            for image in images:
                encoded = self._encode_image(image)
                content.append({
                    "type": "image",
                    "image_base64": encoded,
                })
            return {"role": role, "content": content}
        return {"role": role, "content": text}

    def call(self, messages: List[Dict], n: int = 1, **kwargs) -> List[str]:
        if not self.is_available():
            logger.error("Local model URL not configured")
            return [""]
        payload = {            
            "messages": messages,      
        }
        try:
            response = requests.post(
                self.base_url,
                json=payload,
            )
            if response.status_code == 200:
                data = response.json()
                output_text = data.get("output_text", [""])
                if isinstance(output_text, list):
                    return output_text if output_text else [""]
                return [str(output_text)] if output_text else [""]
            else:
                logger.error(f"Local model API error: {response.status_code}")
                return [""]

        except Exception as e:
            logger.error(f"Local model API call failed: {e}")
            return [""]


