"""
Grounding Agent - 将文本描述转换为屏幕坐标
支持 LocalModelClient 和 OpenAIClient
"""

import base64
import logging
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from mm_agents.llm_server import OpenAIClient, LocalModelClient, LLMClient
from mm_agents.prompts import GROUNDING_SYSTEM_PROMPT, PHRASE_TO_WORD_COORDS_PROMPT


logger = logging.getLogger("desktopenv.agent.grounding")


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


class GroundingAgent:
    
    
    def __init__(
        self,
        client_type: Literal["local", "openai"] = "local",
        model: str = "grounding",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.client_type = client_type
        self.model = model
        self.base_url = base_url or os.environ.get("GROUNDING_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.width = width
        self.height = height
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client: LLMClient = self._create_client()
        logger.info(f"GroundingAgent initialized: {client_type}, {model}")
    
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
    
    def ground(self, reference: str, obs: Dict, custom_prompt: Optional[str] = None) -> Optional[Tuple[int, int]]:
        try:
            screenshot = obs.get("screenshot")
            if screenshot is None:
                return None
            
            if isinstance(screenshot, bytes):
                image_pil = Image.open(BytesIO(screenshot))
                W, H = image_pil.size
                image_base64 = encode_image(screenshot)
            else:
                image_base64 = screenshot
                W, H = self.width, self.height
            
            system_prompt = custom_prompt or self.GROUNDING_SYSTEM_PROMPT.format(width=W, height=H)
            messages = self.client.build_messages(
                system_prompt=system_prompt,
                user_prompt=f"Find the element: {reference}",
                images=[image_base64],
            )
            
            responses = self.client.call(messages)
            result = responses[0] if responses else ""
            
            if result:
                coords = self._parse_coordinates(result, W, H)
                if coords:
                    return coords
            return None
            
        except Exception as e:
            logger.error(f"Grounding error: {e}")
            return None
    
    def ground_normalized(self, reference: str, obs: Dict, custom_prompt: Optional[str] = None) -> Tuple[float, float]:
        coords = self.ground(reference, obs, custom_prompt)
        if coords:
            return coords[0] / self.width, coords[1] / self.height
        return 0.5, 0.5
    
    def ground_text_phrase(self, phrase: str, obs: Dict, alignment: str = "center") -> Tuple[int, int]:
        screenshot = obs.get("screenshot")
        if screenshot is None:
            return self.width // 2, self.height // 2
        
        ocr_table, ocr_elements = self._get_ocr_elements(screenshot)
        if not ocr_elements:
            return self.width // 2, self.height // 2
        
        text_id = self._find_text_match(phrase, ocr_elements)
        if text_id is not None and text_id < len(ocr_elements):
            return self._get_element_coords(ocr_elements[text_id], alignment)
        return self.width // 2, self.height // 2
    
    def ground_with_ocr_context(self, phrase: str, obs: Dict, alignment: str = "center") -> Tuple[int, int]:
        screenshot = obs.get("screenshot")
        if screenshot is None:
            return self.width // 2, self.height // 2
        
        ocr_table, ocr_elements = self._get_ocr_elements(screenshot)
        if not ocr_elements:
            coords = self.ground(phrase, obs)
            return coords if coords else (self.width // 2, self.height // 2)
        
        alignment_prompt = ""
        if alignment == "start":
            alignment_prompt = "Output the FIRST word id.\n"
        elif alignment == "end":
            alignment_prompt = "Output the LAST word id.\n"
        
        user_prompt = f"{alignment_prompt}Phrase: {phrase}\n\nText:\n{ocr_table}\n\nOutput word id:"
        
        try:
            image_base64 = encode_image(screenshot) if isinstance(screenshot, bytes) else screenshot
            messages = self.client.build_messages(
                system_prompt=self.PHRASE_TO_WORD_COORDS_PROMPT,
                user_prompt=user_prompt,
                images=[image_base64],
            )
            
            responses = self.client.call(messages)
            result = responses[0] if responses else ""
            word_id = self._parse_word_id(result)
            
            if word_id is not None and word_id < len(ocr_elements):
                return self._get_element_coords(ocr_elements[word_id], alignment)
        except Exception as e:
            logger.error(f"OCR grounding error: {e}")
        
        return self.ground_text_phrase(phrase, obs, alignment)
    
    def ground_batch(self, references: List[str], obs: Dict) -> List[Optional[Tuple[int, int]]]:
        return [self.ground(ref, obs) for ref in references]
    
    def _get_ocr_elements(self, image_data: bytes) -> Tuple[str, List[Dict]]:
        if not PYTESSERACT_AVAILABLE:
            return "", []
        
        try:
            image = Image.open(BytesIO(image_data))
            ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
            
            for i, word in enumerate(ocr_data["text"]):
                ocr_data["text"][i] = re.sub(r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word)
            
            ocr_elements = []
            ocr_table = "Word id\tText\n"
            grouping_map = defaultdict(list)
            ocr_id = 0
            
            for i in range(len(ocr_data["text"])):
                if ocr_data["text"][i]:
                    block_num = ocr_data["block_num"][i]
                    grouping_map[block_num].append(ocr_data["text"][i])
                    ocr_table += f"{ocr_id}\t{ocr_data['text'][i]}\n"
                    ocr_elements.append({
                        "id": ocr_id, "text": ocr_data["text"][i],
                        "left": ocr_data["left"][i], "top": ocr_data["top"][i],
                        "width": ocr_data["width"][i], "height": ocr_data["height"][i],
                    })
                    ocr_id += 1
            return ocr_table, ocr_elements
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", []
    
    def _parse_coordinates(self, response: str, width: int, height: int) -> Optional[Tuple[int, int]]:
        try:
            patterns = [
                r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)',
                r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]',
                r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    if x <= 1.0 and y <= 1.0:
                        x, y = int(x * width), int(y * height)
                    else:
                        x, y = int(x), int(y)
                    return (max(0, min(x, width)), max(0, min(y, height)))
            return None
        except:
            return None
    
    def _parse_word_id(self, response: str) -> Optional[int]:
        try:
            numbers = re.findall(r'\d+', response)
            return int(numbers[-1]) if numbers else None
        except:
            return None
    
    def _find_text_match(self, phrase: str, elements: List[Dict]) -> Optional[int]:
        phrase_lower = phrase.lower()
        for elem in elements:
            if elem["text"].lower() == phrase_lower:
                return elem["id"]
        for elem in elements:
            if elem["text"].lower() in phrase_lower:
                return elem["id"]
        for word in phrase_lower.split():
            for elem in elements:
                if elem["text"].lower() == word:
                    return elem["id"]
        return None
    
    def _get_element_coords(self, elem: Dict, alignment: str = "center") -> Tuple[int, int]:
        if alignment == "start":
            return elem["left"], elem["top"] + elem["height"] // 2
        elif alignment == "end":
            return elem["left"] + elem["width"], elem["top"] + elem["height"] // 2
        return elem["left"] + elem["width"] // 2, elem["top"] + elem["height"] // 2
    
    def _encode_numpy_image(self, image: np.ndarray) -> str:
        if CV2_AVAILABLE:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.png', image)
            if success:
                return base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            pil_image = Image.fromarray(image)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        return ""
