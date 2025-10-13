import base64
import json
import logging
import os
import re
import time
import ast
import textwrap
from io import BytesIO
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import openai
import requests
import pytesseract
import numpy as np
import cv2
from PIL import Image
from pytesseract import Output
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from requests.exceptions import SSLError
from google.api_core.exceptions import (
    InvalidArgument,
    ResourceExhausted,
    InternalServerError,
    BadRequest,
)

from mm_agents.prompts import GTA1_PLANNER_SYSTEM_PROMPT, GTA1_GROUNDING_SYSTEM_PROMPT
from mm_agents.prompts import MGA_OBSERVATION_PROMPT

logger = logging.getLogger("desktopenv.agent.run")

os.environ["OPENAI_API_KEY"] = ""
proxies = None  # Your proxies
grounding_url = ""
space_url = ""
base_url = ""

def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")


class LMMEngineOpenAI:
    '''
    functions borrow from https://github.com/simular-ai/Agent-S/blob/main/gui_agents/s2/core/engine.py#L247
    '''
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )

        self.base_url = base_url

        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        if not self.base_url:
            self.llm_client = OpenAI(api_key=self.api_key)
        else:
            self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_new_tokens if max_new_tokens else 4096,
                **kwargs,
            )
            .choices[0]
            .message.content
        )

class LMMAgent:
    '''
    functions borrow from https://github.com/simular-ai/Agent-S/blob/a0c5c9bf0c526119b1f023c8948563c780729428/gui_agents/s2/core/mllm.py#L16
    '''
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")
                if engine_type == "openai":
                    self.engine = LMMEngineOpenAI(**engine_params)
                else:
                    raise ValueError("engine_type is not supported")
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages = []

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def encode_image(self, image_content):
        # if image_content is a path to an image file, check type of the image_content to verify
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return base64.b64encode(image_content).decode("utf-8")

    def reset(
        self,
    ):

        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index):
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self, index, text_content, image_content=None, image_detail="high"
    ):
        """Replace a message at a given index"""
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }
            if image_content:
                base64_image = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

    def add_message(
        self,
        text_content,
        image_content=None,
        role=None,
        image_detail="high",
        put_text_last=False,
    ):
        """Add a new message to the list of messages"""

        # API-style inference from OpenAI and AzureOpenAI
        if isinstance(
            self.engine,
            (
                LMMEngineOpenAI,
            ),
        ):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role,
                "content": [{"type": "text", "text": text_content}],
            }

            if isinstance(image_content, np.ndarray) or image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        }
                    )

            # Rotate text to be the last message if desired
            if put_text_last:
                text_content = message["content"].pop(0)
                message["content"].append(text_content)

            self.messages.append(message)
        else:
            raise ValueError("engine_type is not supported")

    def get_response(
        self,
        user_message=None,
        messages=None,
        temperature=0.0,
        max_new_tokens=None,
        **kwargs,
    ):
        """Generate the next response based on previous messages"""
        if messages is None:
            messages = self.messages
        if user_message:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )

        return self.engine.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


UBUNTU_APP_SETUP = """import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""


SET_CELL_VALUES_CMD = """import uno
import subprocess

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"password\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""

    
class OSWorldACI:
    '''
    classes borrow from https://github.com/simular-ai/Agent-S/blob/a0c5c9bf0c526119b1f023c8948563c780729428/gui_agents/s2/agents/grounding.py#L159
    '''
    PHRASE_TO_WORD_COORDS_PROMPT = textwrap.dedent(
        """
    You are an expert in graphical user interfaces. Your task is to process a phrase of text, and identify the most relevant word on the computer screen.
    You are provided with a phrase, a table with all the text on the screen, and a screenshot of the computer screen. You will identify the single word id that is best associated with the provided phrase.
    This single word must be displayed on the computer screenshot, and its location on the screen should align with the provided phrase.
    Each row in the text table provides 2 pieces of data in the following order. 1st is the unique word id. 2nd is the corresponding word.

    To be successful, it is very important to follow all these rules:
    1. First, think step by step and generate your reasoning about which word id to click on.
    2. Then, output the unique word id. Remember, the word id is the 1st number in each row of the text table.
    3. If there are multiple occurrences of the same word, use the surrounding context in the phrase to choose the correct one. Pay very close attention to punctuation and capitalization.

    """
    )
    def __init__(
        self,
        platform: 'linux',
        width: int = 1920,
        height: int = 1080,
        _logger=None
    ):
        self.platform = (
            platform  # Dictates how the switch_applications agent action works.
        )
        
        engine_params_for_generation = engine_params = {
            "engine_type": 'openai',
            "model": 'o3',
            "base_url": '',
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }
        
        # Configure scaling
        self.width = width
        self.height = height

        # Maintain state for save_to_knowledge
        self.notes = []

        # Coordinates used during ACI execution
        self.coords1 = None
        self.coords2 = None

        # Configure text grounding agent
        self.text_span_agent = LMMAgent(
            engine_params=engine_params_for_generation,
            system_prompt=self.PHRASE_TO_WORD_COORDS_PROMPT,
        )
        self.runtime_logger = _logger

    # Given the state and worker's referring expression, use the grounding model to generate (x,y)
    def generate_coords(self, ref_expr: str, obs: Dict, request_vllm) -> List[int]:
        return request_vllm(image=obs["screenshot"], prompt=ref_expr)

    # Calls pytesseract to generate word level bounding boxes for text grounding
    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        image = Image.open(BytesIO(b64_image_data))
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Clean text by removing leading and trailing spaces and non-alphabetical characters, but keeping punctuation
        for i, word in enumerate(image_data["text"]):
            image_data["text"][i] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word
            )
        ocr_elements = []
        ocr_table = "Text Table:\nWord id\tText\n"
        # Obtain the <id, text, group number, word number> for each valid element
        grouping_map = defaultdict(list)
        ocr_id = 0
        for i in range(len(image_data["text"])):
            block_num = image_data["block_num"][i]
            if image_data["text"][i]:
                grouping_map[block_num].append(image_data["text"][i])
                ocr_table += f"{ocr_id}\t{image_data['text'][i]}\n"
                ocr_elements.append(
                    {
                        "id": ocr_id,
                        "text": image_data["text"][i],
                        "group_num": block_num,
                        "word_num": len(grouping_map[block_num]),
                        "left": image_data["left"][i],
                        "top": image_data["top"][i],
                        "width": image_data["width"][i],
                        "height": image_data["height"][i],
                    }
                )
                ocr_id += 1

        return ocr_table, ocr_elements

    # Given the state and worker's text phrase, generate the coords of the first/last word in the phrase
    def generate_text_coords(
        self, phrase: str, obs: Dict, alignment: str = ""
    ) -> List[int]:
        ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])

        alignment_prompt = ""
        if alignment == "start":
            alignment_prompt = "**Important**: Output the word id of the FIRST word in the provided phrase.\n"
        elif alignment == "end":
            alignment_prompt = "**Important**: Output the word id of the LAST word in the provided phrase.\n"
            
        # Load LLM prompt
        self.text_span_agent.reset()
        self.text_span_agent.add_message(
            alignment_prompt + "Phrase: " + phrase + "\n" + ocr_table, role="user"
        )
        self.text_span_agent.add_message(
            "Screenshot:\n", image_content=obs["screenshot"], role="user"
        )

        # Obtain the target element
        response = call_llm_safe(self.text_span_agent)
        numericals = re.findall(r"\d+", response)
        if len(numericals) > 0:
            text_id = int(numericals[-1])
        else:
            text_id = 0
        elem = ocr_elements[text_id]

        # Compute the element coordinates
        if alignment == "start":
            coords = [elem["left"], elem["top"] + (elem["height"] // 2)]
        elif alignment == "end":
            coords = [elem["left"] + elem["width"], elem["top"] + (elem["height"] // 2)]
        else:
            coords = [
                elem["left"] + (elem["width"] // 2),
                elem["top"] + (elem["height"] // 2),
            ]
        return coords

    # Takes a description based action and assigns the coordinates for any coordinate based action
    # Raises an error if function can't be parsed
    def assign_coordinates(self, plan: str, obs: Dict, request_vllm):

        # Reset coords from previous action generation
        self.coords1, self.coords2 = None, None

        try:
            # Extract the function name and args
            action = parse_single_code_from_string(plan.split("Grounded Action")[-1])
            
            # Clean action string, remove comments and escape characters
            action_lines = action.split('\n')
            clean_action = ""
            for line in action_lines:
                line = line.strip()
                if line and not line.startswith('#') and line.startswith('agent.'):
                    clean_action = line
                    break
        
            if not clean_action:
                clean_action = action
                
            function_name = re.match(r"(\w+\.\w+)\(", clean_action).group(1)
            args = self.parse_function_args(clean_action)
        except Exception as e:
            raise RuntimeError(f"Error in parsing grounded action: {e}") from e

        # arg0 is a description
        if (
            function_name in ["agent.click", "agent.double_click", "agent.type", "agent.scroll"]
            and len(args) >= 1
            and args[0] != None
        ):
            self.coords1 = self.generate_coords(args[0], obs, request_vllm)
        # arg0 and arg1 are descriptions
        elif function_name == "agent.drag_and_drop" and len(args) >= 2:
            self.coords1 = self.generate_coords(args[0], obs, request_vllm)
            self.coords2 = self.generate_coords(args[1], obs, request_vllm)
        # arg0 and arg1 are text phrases
        elif function_name == "agent.highlight_text_span" and len(args) >= 2:
            self.coords1 = self.generate_text_coords(args[0], obs, alignment="start")
            self.coords2 = self.generate_text_coords(args[1], obs, alignment="end")          

    # Resize from grounding model dim into OSWorld dim (1920 * 1080)
    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        return [
            round(coordinates[0] * self.width),
            round(coordinates[1] * self.height),
        ]

    # Given a generated ACI function, returns a list of argument values, where descriptions are at the front of the list
    def parse_function_args(self, function: str) -> List[str]:
        tree = ast.parse(function)
        call_node = tree.body[0].value

        def safe_eval(node):
            if isinstance(
                node, ast.Constant
            ):  # Handles literals like numbers, strings, etc.
                return node.value
            else:
                return ast.unparse(node)  # Return as a string if not a literal

        positional_args = [safe_eval(arg) for arg in call_node.args]
        keyword_args = {kw.arg: safe_eval(kw.value) for kw in call_node.keywords}

        res = []

        for key, val in keyword_args.items():
            if "description" in key:
                res.append(val)

        for arg in positional_args:
            res.append(arg)

        return res

    def click(
        self,
        instruction: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            instruction:str, decribe the element you want to interact with in detail including the visual description and function description. And make it clear and concise. For example you can describe what the element looks like, and what will be the expected result when you interact with it.
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        x, y = self.resize_coordinates(self.coords1)
        command = "import pyautogui; "

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    def double_click(
        self,
        instruction: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            instruction:str, decribe the element you want to interact with in detail including the visual description and function description. And make it clear and concise. For example you can describe what the element looks like, and what will be the expected result when you interact with it.
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        x, y = self.resize_coordinates(self.coords1)
        command = "import pyautogui; "

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks*2}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    def switch_applications(self, app_code):
        """Switch to a different application that is already open
        Args:
            app_code:str the code name of the application to switch to from the provided list of open applications
        """
        if self.platform == "darwin":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "linux":
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_code)
        elif self.platform == "windows":
            return f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"

    def open(self, app_or_filename: str):
        """Open any application or file with name app_or_filename. Use this action to open applications or files on the desktop, do not open manually.
        Args:
            app_or_filename:str, the name of the application or filename to open
        """
        return f"import pyautogui; pyautogui.hotkey('win'); time.sleep(0.5); pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); pyautogui.hotkey('enter'); time.sleep(0.5)"

    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into a specific element
        Args:
            element_description:str, a detailed description of which element to enter text in. This description should be at least a full sentence.
            text:str, the text to type
            overwrite:bool, Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool, Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """

        if self.coords1 is not None:
            # If a node is found, retrieve its coordinates and size
            # Start typing at the center of the element

            x, y = self.resize_coordinates(self.coords1)

            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # If no element is found, start typing at the current cursor location
            command = "import pyautogui; "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "

        return command

    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """Drag from the starting description to the ending description
        Args:
            starting_description:str, a very detailed description of where to start the drag action. This description should be at least a full sentence. And make it clear and concise.
            ending_description:str, a very detailed description of where to end the drag action. This description should be at least a full sentence. And make it clear and concise.
            hold_keys:List list of keys to hold while dragging
        """
        x1, y1 = self.resize_coordinates(self.coords1)
        x2, y2 = self.resize_coordinates(self.coords2)

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    def highlight_text_span(self, starting_phrase: str, ending_phrase: str):
        """Highlight a text span between a provided starting phrase and ending phrase. Use this to highlight words, lines, and paragraphs.
        Args:
            starting_phrase:str, the phrase that denotes the start of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
            ending_phrase:str, the phrase that denotes the end of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
        """

        x1, y1 = self.coords1
        x2, y2 = self.coords2

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "

        # Return pyautoguicode to drag and drop the elements
        return command

    def set_cell_values(
        self, cell_values: Dict[str, Any], app_name: str, sheet_name: str
    ):
        """Use this to set individual cell values in a spreadsheet. For example, setting A2 to "hello" would be done by passing {"A2": "hello"} as cell_values. The sheet must be opened before this command can be used.
        Args:
            cell_values: Dict[str, Any], A dictionary of cell values to set in the spreadsheet. The keys are the cell coordinates in the format "A1", "B2", etc.
                Supported value types include: float, int, string, bool, formulas.
            app_name: str, The name of the spreadsheet application. For example, "Some_sheet.xlsx".
            sheet_name: str, The name of the sheet in the spreadsheet. For example, "Sheet1".
        """
        return SET_CELL_VALUES_CMD.format(
            cell_values=cell_values, app_name=app_name, sheet_name=sheet_name
        )

    def scroll(self, instruction: str, clicks: int, shift: bool = False):
        """Scroll the element in the specified direction
        Args:
            instruction:str, a very detailed description of which element to enter scroll in. This description should be at least a full sentence. And make it clear and concise.
            clicks:int, the number of clicks to scroll can be positive (up) or negative (down).
            shift:bool, whether to use shift+scroll for horizontal scrolling
        """

        x, y = self.resize_coordinates(self.coords1)

        if shift:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.hscroll({clicks})"
        else:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.vscroll({clicks})"

    def hotkey(self, keys: List):
        """Press a hotkey combination
        Args:
            keys:List the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
        """
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"""import time; time.sleep({time})"""

    def done(
        self,
        return_value: Optional[Union[Dict, str, List, Tuple, int, float, bool]] = None,
    ):
        """End the current task with a success and the required return value"""
        self.returned_info = return_value
        return """DONE"""

    def fail(self):
        """End the current task with a failure, and replan the whole task."""
        return """FAIL"""

def call_llm_safe(agent):
    """Safely call LLM with retry mechanism"""
    max_retries = 3
    attempt = 0
    response = ""
    while attempt < max_retries:
        try:
            response = agent.get_response()
            break
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
    return response

def parse_single_code_from_string(input_string):
    """Parse single code block from string"""
    input_string = input_string.strip()
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return input_string.strip()

    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)
    codes = []

    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL"]

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes[-1]

agent = OSWorldACI('linux')

class MGA1Agent:
    '''
    class based on https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/jedi_7b_agent.py
    '''
    def __init__(
        self,
        platform="ubuntu",
        planner_model="o3",
        max_tokens=4096,
        top_p=0.9,
        temperature= 0.0,
        action_space="pyautogui",
        observation_type="screenshot",
        max_steps=30,
        max_image_history_length = 5,
        N_SEQ = 1,
        client_password="password"
    ):
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.client_password = client_password
        self.observation_type = observation_type
        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.observation_captions = []
        self.memory = []
        self.max_steps = max_steps
        self.planner_model=planner_model
        self.summary_model="o3"
        self.current_step = 0
        self.max_image_history_length = max_image_history_length
        self.N_SEQ=1


    def detailed_observation_via_grounding(self, obs: Dict) -> str:
        """
        通过grounding_url获取详细的观测信息
        """
        try:
            image_base64 = encode_image(obs['screenshot'])
            
            payload = {
                "system_text": [MGA_OBSERVATION_PROMPT],
                "user_text": ["Please provide a detailed analysis of the current screenshot focusing on spatial layout, semantic meaning, and interactive elements."],
                "assistant_text": [""],
                "user_image_base64": [image_base64]
            }
            
            response = requests.post(space_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                data = data["output_text"][0] if isinstance(data["output_text"], list) and len(data["output_text"]) > 0 else ""
                return data
            else:
                logger.error(f"本地模型API错误: {response.status_code} - {response.text}")
                return ""

                
        except Exception as e:
            logger.error(f"Error during grounding observation: {str(e)}")
            return "Error occurred during detailed observation phase."

    def summarize_with_o3(self, instruction: str, last_thought : str,last_actions : str,last_observation : str, last_memory: str) -> str:

        summary_prompt = """
You are an expert at summarizing GUI automation contexts. Your task is to analyze the sequence of past observations, thoughts, and actions, and provide a concise summary of what has happened so far. Do not provide any suggestions or recommendations for the next step; just summarize the previous operations to serve as a reference for further reasoning.
"""

        user_summary_prompt = f"""
**Current Task:** {instruction}

**Previous Memory Context:**
{last_memory}

**Latest Step Analysis:**
- **Last Thought:** {last_thought}
- **Last Action:** {last_actions}
- **Last Observation:** {last_observation}

**Memory Integration and State Transition Summary:**

Based on the accumulated historical memory and the most recent step execution, provide a comprehensive abstract summary that captures the following aspects:

1. **Interface State Evolution:**
   - Describe how the GUI state has transformed from the historical context through the latest operation.
   - Document specific changes in windows, applications, UI elements, and their current status, ensuring to include exact names of files, paths, or objects involved.
   - Record any state transitions, mode changes, or workflow progressions that have occurred, explicitly mentioning the names of affected elements.
   - Note the current interface configuration compared to previous states, including detailed references to specific files, directories, or UI components.

2. **Operation Effect Analysis:**
   - Analyze what concrete effects the latest action produced on the system, explicitly mentioning the names of files, paths, or objects affected.
   - Determine whether the operation achieved its intended outcome or resulted in unexpected behavior, with detailed references to the specific elements involved.
   - Assess if the interface responded in a logically consistent manner to the executed action, referencing the exact objects or elements interacted with.
   - Document any side effects or secondary changes that occurred, ensuring to include specific details about the affected files, paths, or objects.

3. **Behavioral Pattern Recognition:**
   - Identify recurring operational patterns across the historical sequence, explicitly referencing the names of files, paths, or objects involved.
   - Detect any repetitive actions that may indicate inefficient or stuck behavior, with detailed references to the specific elements affected.
   - Recognize successful operation chains that achieved desired state changes, explicitly mentioning the names of files, paths, or objects involved.
   - Note any deviations from established interaction patterns, ensuring to include specific details about the affected elements.

4. **Issue Identification and Classification:**
   - **Redundant Operations:** Identify actions that appear unnecessary or duplicative within the sequence, explicitly referencing the names of files, paths, or objects involved.
   - **Erroneous Operations:** Detect actions that seem to contradict the task objectives or expected workflow, with detailed references to the specific elements affected.
   - **State Inconsistencies:** Flag any unexplained state changes or interface responses that don't align with the performed actions, explicitly mentioning the names of affected files, paths, or objects.
   - **Efficiency Concerns:** Note operational sequences that could potentially be optimized or streamlined, ensuring to include specific details about the affected elements.

5. **State Consistency Verification:**
   - Evaluate whether the current state logically follows from the sequence of executed operations, explicitly referencing the names of files, paths, or objects involved.
   - Identify any gaps or inconsistencies in the state transition chain, ensuring to include specific details about the affected elements.
   - Assess the overall coherence of the operation-to-state mapping, explicitly mentioning the names of files, paths, or objects involved.
   - Document any anomalous system behaviors or unexpected interface responses, ensuring to include specific details about the affected elements.

**Summary Output Requirements:**
This summary should serve as a detailed observational record focusing on state transitions and issue identification. Avoid any operational recommendations, strategic suggestions, or next-step guidance. The output will function as a memory artifact for tracking task execution patterns and maintaining awareness of potential operational concerns without influencing future decision-making processes.
"""
        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": summary_prompt
            }]
        }]

        messages.append({
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": user_summary_prompt
                },
            ],
        })

        summaries_response_ = self.call_llm({
            "model": self.summary_model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }, self.summary_model)

        return summaries_response_
        



    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """


        if len(self.thoughts) > 0:
            o3_summary = self.summarize_with_o3(instruction, self.thoughts[-1],self.actions[-1], self.observation_captions[-1], self.memory[-1])

        else:
            o3_summary = "No previous steps."

        logger.debug(f"\nStep {self.current_step + 1} summary : {o3_summary}")

        self.memory.append(o3_summary)

        user_prompt = (
                    f"""Please generate the next move according to the UI screenshot, Detailed Current Observation and instruction. And you can refer to the summary of previous actions and observations for reflection.

        **Previous Summary:**
        {o3_summary}

        **Instruction:** {instruction}

        Your primary focus should be on making decisions based on the Detailed Current Observation, the Screen Spatial Analysis and the Instruction provided. These are the key elements guiding your actions.

        Additionally, you should carefully review the Potential Issues highlighted in the Previous Summary to avoid repeating mistakes or encountering similar obstacles. Use this information as a reference to refine your approach, but always prioritize the current observation and instruction when determining the next steps.

        """)

        system_prompt = GTA1_PLANNER_SYSTEM_PROMPT

        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt.replace("{current_step}", str(self.current_step)).replace("{max_steps}", str(self.max_steps))
            }]
        }]

        # Determine which observations to include images for (only most recent ones)
        obs_start_idx = max(0, len(self.observations) - self.max_image_history_length)

        messages.append({
                "role":"user",
                "content": [
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/png;base64,{encode_image(obs['screenshot'])}",
                            "detail": "high"
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                ],
            })


        N = self.N_SEQ
        planner_response = []

        for bn in split_to_batches(N, batch_size=8):
            planner_response_ = self.call_llm({
                    "model": self.planner_model,
                    "messages": messages,
                    "n": bn,
                    "max_completion_tokens": self.max_tokens,
                }, self.planner_model)
            planner_response.extend(planner_response_)

        valid_responses = [response for response in planner_response if self.isvalid(response)]

        N = N - len(valid_responses)
        planner_response = [response for response in planner_response if not self.isvalid(response)]
        if planner_response:
            planner_response = planner_response[0]
        retry_count = 0
        max_retries = 5
        
        
        while N > 0: 
            if retry_count >= max_retries:
                break
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": """You didn't generate a valid "Observation:\n(.*?)\n" section, a valid "Thought:\n(.*?)\n" section,  or valid actions. Please try again."""}
                ]
            })
                
            planner_response = []
            for bn in split_to_batches(N, batch_size=8):
                planner_response_ = self.call_llm({
                        "model": self.planner_model,
                        "messages": messages,
                        "n": bn,
                        "max_completion_tokens": self.max_tokens * 4,
                    }, self.planner_model)
                planner_response.extend(planner_response_)

            valid_responses_ = [response for response in planner_response if self.isvalid(response)]
            N = N - len(valid_responses_)
            planner_response = [response for response in planner_response if not self.isvalid(response)]
            if planner_response:
                planner_response = planner_response[0]
            valid_responses.extend(valid_responses_)
            retry_count += 1
        
        assert len(valid_responses) > int(self.N_SEQ) * 0.5, f"Not enough valid responses generated {len(valid_responses)}"
        if self.N_SEQ > 1:
            history_cache = [f"Observation:\n{o}\nThought:\n{t}\nAction:\n{a}" for a,t,o in zip(self.actions, self.thoughts, self.observations)]
            planner_response = self.select(instruction, Image.open(BytesIO(obs['screenshot'])), valid_responses, history_cache)
        else:
            planner_response = valid_responses[0]
        codes = self.parse_code_from_planner_response(planner_response)
        thought = self.parse_thought_from_planner_response(planner_response)
        observation_caption = self.parse_observation_caption_from_planner_response(planner_response)
        
        def request_vllm(image, prompt):
            if isinstance(image, bytes):
                image = np.array(Image.open(BytesIO(image)).convert('RGB'))
            H, W, C = image.shape
            assert C == 3
            if isinstance(image, np.ndarray):
                image_base64 = encode_numpy_image_to_base64(image)
            elif isinstance(image, bytes):
                image_base64 = encode_image_bytes(image)
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": GTA1_GROUNDING_SYSTEM_PROMPT.format(height=H, width=W)
                        }
                    ]
                },    
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }]

            
            system_text = []
            user_text = []
            user_image_base64 = []
            
            for msg in messages:
                if msg["role"] == "system":
                    for content in msg["content"]:
                        if content["type"] == "text":
                            system_text.append(content["text"] + "\n")
            
            for msg in reversed(messages):
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "text":
                            user_text.append(content["text"])
                        elif content["type"] == "image_url":
                            # 提取base64图像数据
                            user_image_base64.append(content["image_url"]["url"].replace("data:image/png;base64,", ""))
                    break
                
            payload = {
                "system_text": system_text,
                "user_text": user_text,
                "assistant_text": [],
                "user_image_base64": user_image_base64
            }   

            try:
                response = requests.post(grounding_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    data = data["output_text"][0] if isinstance(data["output_text"], list) and len(data["output_text"]) > 0 else ""
                else:
                    logger.error(f"本地模型API错误: {response.status_code} - {response.text}")
                    return ""
            except Exception as e:
                logger.error(f"调用本地模型时出错: {str(e)}")
                return ""

            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", data)
            x,y =  [tuple(map(int, match)) for match in matches][0]
            x = x/W
            y = y/H
            return x,y
        
        agent.assign_coordinates(planner_response, obs, request_vllm)
        if isinstance(codes, list):
            plan_code = extract_first_agent_function(codes[-1])
        else:
            plan_code = extract_first_agent_function("".join(codes))


        try:
            plan_code = plan_code.replace('\n', ' ')
            pyautogui_actions = [eval(plan_code)]
        except Exception as e:
            logger.debug(f"Error evaluating plan_code: {plan_code}, error: {e}")

        plan_code = [plan_code]
        self.actions.append([plan_code])
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.observation_captions.append(observation_caption)
        self.current_step += 1
     
        if self.current_step >= self.max_steps:
            pyautogui_actions = ["FAIL"]

        logger.debug(f"\nStep {self.current_step}: previous summary - {o3_summary}")
        logger.debug(f"\nStep {self.current_step}: Thoughts - {thought}")
        logger.debug(f"\nStep {self.current_step}: Codes - {plan_code}")
        logger.debug(f"\nStep {self.current_step}: Actions - {pyautogui_actions}")

        return planner_response, pyautogui_actions 
    
    def isvalid(self,planner_response):
        codes = self.parse_code_from_planner_response(planner_response)
        thought = self.parse_thought_from_planner_response(planner_response)
        observation_caption = self.parse_observation_caption_from_planner_response(planner_response)
        return bool(codes and thought and observation_caption)
    
    def parse_code_from_planner_response(self, input_string: str) -> List[str]:

        input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
        
        pattern = r"```(?:\w+\s+)?(.*?)```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        codes = []

        for match in matches:
            match = match.strip()
            codes.append(match)
        return codes
    
    def unsetonestep(self):
        self.actions = self.actions[:-1]
        self.observations = self.observations[:-1]
        self.thoughts = self.thoughts[:-1]
        self.observation_captions = self.observation_captions[:-1]
        self.current_step -= 1
        
    def parse_observation_caption_from_planner_response(self, input_string: str) -> str:
        pattern = r"Observation:\n(.*?)\n"
        matches = re.findall(pattern, input_string, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    def parse_thought_from_planner_response(self, input_string: str) -> str:
        # pattern = r"Thought:\n(.*?)\n"
        pattern = r"Thought:(.*?)\n(?:Action:|Step|```)"
        matches = re.findall(pattern, input_string, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""
        
    @backoff.on_exception(
        backoff.constant,
        (
            SSLError,
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
        ),
        interval=30,
        max_tries=10,
    )

    def call_llm(self, payload, model):
        if model.startswith("gpt") or "o3" in model:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
            response = requests.post(
                base_url,
                headers=headers,
                proxies=proxies,
                json=payload,
            )
            if response.status_code != 200:
                time.sleep(5)
                return ""
            else:
                response = response.json()
                return [response["choices"][i]["message"]["content"] for i in range(len(response["choices"]))]
        else:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                time.sleep(5)
                return ""
            else:
                response = response.json()
                return [response["choices"][i]["message"]["content"] for i in range(len(response["choices"]))]

    def reset(self, _logger=None):
        global logger

        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent.run1")
        self.current_step = 0
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.observation_captions = []
        self.memory = []
        
        
def extract_first_agent_function(code_string):
    """Extract the first agent function call from a code string"""
    pattern = r'agent\.[a-zA-Z_]+\((?:[^()]|(?:\([^\)]*\))|\'(?:\\.|[^\'])*\'|"(?:\\.|[^"])*")*\)'
    matches = re.findall(pattern, code_string)
    return matches[0] if matches else None
        
def split_to_batches(n, batch_size=8):
    """Split n into batches of given size"""
    batches = [batch_size] * (n // batch_size)
    remainder = n % batch_size
    if remainder:
        batches.append(remainder)
    return batches

def encode_numpy_image_to_base64(image: np.ndarray) -> str:
    """Converts a numpy array image to base64 string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to png format")
    
    image_bytes = buffer.tobytes()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string

def encode_image_bytes(image_content):
    """Encode image bytes to base64"""
    return base64.b64encode(image_content).decode('utf-8')