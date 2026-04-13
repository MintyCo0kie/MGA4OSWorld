"""
Observer Agent - 屏幕观察和分析
支持 LocalModelClient 和 OpenAIClient
"""

import base64
import logging
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from mm_agents.llm_server import OpenAIClient, LocalModelClient, LLMClient

logger = logging.getLogger("desktopenv.agent.observer")


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


MGA_OBSERVATION_PROMPT = """# Role: GUI Screen Observer
You are a precise screen state reader. Your job is to faithfully extract what is currently visible on the screen — which applications are active, what text content is shown, and what the current interactive state is.

Do NOT suggest next steps. Do NOT infer user intent. Report only what is **directly visible**.

---

## Observation Dimensions

### 1. Application & Window State
*What applications and windows are currently open and visible?*
- List all visible application windows with their titles (e.g., "Firefox — Stack Overflow", "Terminal — bash", "LibreOffice Calc — budget.xlsx").
- Identify the **foreground (focused) window** — the one receiving keyboard input.
- Note any **overlapping windows**, **minimized** indicators on the taskbar, or **system dialogs** (e.g., authentication prompt, file picker, confirmation dialog).
- If a **modal dialog** or **overlay** is blocking the main window, describe it explicitly: its title, message text, and visible buttons.

### 2. Visible Text Content
*What text is currently readable on screen? Transcribe it faithfully.*
- **Document / Editor content**: Transcribe visible text in text editors, spreadsheets, terminals, or browsers. For spreadsheets, note visible cell values and column/row headers. For terminals, transcribe the last visible command and its output verbatim.
- **UI Labels & Menus**: List visible menu items, tab names, toolbar labels, and sidebar entries that are currently expanded or visible.
- **Notifications & Alerts**: Transcribe any pop-up messages, toast notifications, error dialogs, or system alerts **word for word**.
- **Form Fields**: Report the current value or placeholder text in input fields, search bars, and address bars.
- **Status Information**: Transcribe status bar content, progress indicators with percentages, or any bottom-bar messages.

### 3. Current Interactive State
*What is the screen's current interaction state?*
- **Cursor / Focus**: Which element currently has keyboard focus (blinking caret, highlighted border)?
- **Selection**: Is any text, file, cell range, or list item currently selected (highlighted)? Describe what is selected.
- **Scroll Position**: Is the content scrolled (e.g., "scrolled to line 45", "bottom of page reached", "showing rows 20–50")?
- **Loading / Processing**: Is anything loading, saving, or running (spinner, progress bar, "Loading…" text)?
- **Transient Overlays**: Are there context menus, autocomplete dropdowns, tooltips, or hover panels currently visible?

---

## Output Format

**1. Application & Window State**
[List all visible windows, identify foreground window, note any blocking dialogs]

**2. Visible Text Content**
[Transcribe terminal output, document content, dialog messages, form values, notifications — verbatim where possible]

**3. Current Interactive State**
[Describe focus, selection, scroll position, loading state, and any transient overlays]
"""


KIMI_SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}



In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}

Do NOT output any numeric coordinates, use ELEMENT DESCRIPTION to replace the coordinates. The element description should be a concise sentence describing the element and its position on the screen,
for example, pyautogui.click("the Save button at the bottom right corner of the Save As dialog", clicks=1, button="left", duration=1).

""".strip()

class ObserverAgent:

    def __init__(
        self,
        client_type: Literal["local", "openai"] = "local",
        model: str = "observer",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.client_type = client_type
        self.model = model
        self.base_url = base_url or os.environ.get("OBSERVER_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client: LLMClient = self._create_client()
        logger.info(f"ObserverAgent initialized: {client_type}, {model}")
    
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
    
    def observe(self, image_base64) -> str:
        try:
            messages = self.client.build_messages(
                system_prompt=MGA_OBSERVATION_PROMPT,
                user_prompt="please analyze this screenshot.",
                images=[image_base64],
            )
            
            obervation = self._call_llm(messages)
            return obervation

        except Exception as e:
            logger.error(f"Observation error: {e}")
            return "Error during observation."

    def _call_llm(self, messages: List[Dict]) -> str:
        responses = self.client.call(messages)
        if self.model.startswith("gpt"):
            return responses if responses else ""
        return responses[0] if responses else ""
    
    def observe_with_prompt(self, image_base64, system_prompt: str, user_prompt: str) -> str:
        try:
            
            messages = self.client.build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=[image_base64],
            )
            
            responses = self.client.call(messages)
            return responses[0] if responses else ""
        except Exception as e:
            logger.error(f"Observation error: {e}")
            return ""
    
    def get_ocr_elements(self, image_data: bytes) -> Tuple[str, List[Dict]]:
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
    