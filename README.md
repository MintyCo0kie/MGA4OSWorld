# MGA for OSworld User Guide

## Project Overview

MGA (Memory-Driven GUI Agent) is one of the key intelligent agent implementations that can complete complex operating system tasks through visual observation and action execution. All of our experience are based on OSWorld, which is an open-source operating system world benchmark platform that supports multiple virtualization environments and intelligent agents(https://github.com/xlang-ai/OSWorld). 

## System Requirements

### Operating System Requirements
- Linux operating system (Ubuntu 18.04+ recommended)
- KVM virtualization support (optional, for better performance)
- Docker environment

### Hardware Requirements
- Memory: 8GB+ recommended
- Storage: At least 20GB available space
- CPU: Support for virtualization instruction sets

## Installation and Configuration

### 1. Check KVM Support (Recommended)

Check if your machine supports KVM:

```bash
egrep -c '(vmx|svm)' /proc/cpuinfo
```

If the return value is greater than 0, the processor supports KVM.

> **Note**: macOS hosts generally do not support KVM. VMware is recommended for macOS users.

### 2. Install Docker

Install Docker according to your system:

- **Linux with GUI**: Refer to [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux/)
- **Windows**: Refer to [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Headless servers**: Refer to [Install Docker Engine](https://docs.docker.com/engine/install/)

### 3. Project Environment Setup

Clone and set up the project:

```bash
git clone https://github.com/xlang-ai/OSWorld.git
cd OSWorld
pip install -r requirements.txt
```

### 4. Virtual Machine Image Preparation

Ensure you have an Ubuntu virtual machine image file. The project uses `.qcow2` format image files, for example:

```
docker_vm_data/Ubuntu.qcow2
```

If you don't have the image file, the virtual machine snapshot will be automatically downloaded on the first run.

## MGA Agent Architecture

### Core Components

1. **LMMEngineOpenAI**: Language model engine supporting OpenAI API-compatible services
2. **LMMAgent**: Base language model agent class
3. **GroundingAgent**: Visual grounding agent responsible for screen element identification
4. **MGA1Agent**: Main multi-modal intelligent agent class

### Initialization Parameters

MGA Agent supports the following main initialization parameters:

- `model`: Model name to use (e.g., "GTA1")
- `observation_type`: Observation type (screenshot, a11y_tree, etc.)
- `action_space`: Action space (pyautogui, etc.)
- `max_tokens`: Maximum generation token count
- `temperature`: Generation temperature parameter

## Usage Guide

### Basic Run Command

Complete command to run MGA Agent with Docker provider:

```bash
python run.py \
  --provider_name docker \
  --path_to_vm /home/chengweihua/Desktop/cwh/code/OSWorld/docker_vm_data/Ubuntu.qcow2 \
  --headless \
  --observation_type screenshot \
  --model MGA1 \
  --sleep_after_execution 5 \
  --max_steps 50 \
  --result_dir ./results
```

### Parameter Descriptions

#### Environment Configuration Parameters
- `--provider_name docker`: Use Docker as virtualization provider
- `--path_to_vm <path>`: Virtual machine image file path
- `--headless`: Run in headless mode (no GUI display)
- `--screen_width 1920`: Screen width (default 1920)
- `--screen_height 1080`: Screen height (default 1080)

#### Observation and Action Configuration
- `--observation_type screenshot`: Observation type
  - `screenshot`: Screen screenshot
  - `a11y_tree`: Accessibility tree
  - `screenshot_a11y_tree`: Screenshot + accessibility tree
  - `som`: Set-of-Mark technique
- `--action_space pyautogui`: Action space (default pyautogui)

#### Model Configuration
- `--model MGA1`: Model name to use
- `--temperature 1.0`: Generation temperature (default 1.0)
- `--top_p 0.9`: Top-p sampling parameter (default 0.9)
- `--max_tokens 4096`: Maximum generation token count

#### Execution Control
- `--sleep_after_execution 5`: Wait time after action execution (seconds)
- `--max_steps 50`: Maximum execution steps
- `--max_trajectory_length 3`: Maximum trajectory length

#### Output Configuration
- `--result_dir ./results`: Result save directory
- `--domain all`: Test domain (default all)

### Code Example

You can also use MGA Agent directly through Python code:

```python
from mm_agents.MGA_Agent import MGA1Agent
from desktop_env.desktop_env import DesktopEnv

# Initialize MGA Agent
agent = MGA1Agent(
    # model="GTA1",
    observation_type="screenshot",
    action_space="pyautogui",
    max_tokens=4096,
    temperature=1.0
)

# Initialize desktop environment
env = DesktopEnv(
    provider_name="docker",
    path_to_vm="/path/to/your/Ubuntu.qcow2",
    action_space=agent.action_space,
    screen_size=(1920, 1080),
    headless=True,
    os_type="Ubuntu",
    require_a11y_tree=True  # if using a11y_tree observation
)

# Reset environment and agent
env.reset()
agent.reset()

# Execute task
instruction = "Please help me open a text editor"
obs = env.get_observation()
response, actions = agent.predict(instruction, obs)

# Execute actions
for action in actions:
    env.step(action)
```

## Observation Space and Action Space

### Observation Space

- `screenshot`: Screenshot of the current screen
- `a11y_tree`: Accessibility tree of the current screen
- `screenshot_a11y_tree`: Combination of screenshot and accessibility tree
- `som`: Set-of-Mark technique with table metadata

### Action Space

- `pyautogui`: Valid Python code using pyautogui library
- `computer_13`: A set of enumerated actions designed by us

### Observation Data Format

Observation data should be maintained as a dictionary:

```python
obs = {
    "screenshot": open("path/to/observation.jpg", 'rb').read(),
    "a11y_tree": "accessibility tree data"
}
```

## Docker VM Management

### Running Experiments

When using Docker provider, add the following parameters:

- `provider_name`: `docker`
- `os_type`: `Ubuntu` or `Windows` (depending on VM's OS)

### Cleaning Docker Containers

If experiments are abnormally interrupted, residual Docker containers may remain. Run the following command to clean up:

```bash
docker stop $(docker ps -q) && docker rm $(docker ps -a -q)
```

## Troubleshooting

### Common Issues

1. **Docker Permission Issues**
   ```bash
   sudo usermod -aG docker $USER
   # Re-login or run
   newgrp docker
   ```

2. **Slow VM Image Download**
   - Please be patient during the first run for VM snapshot download
   - Ensure stable network connection

3. **Insufficient Memory**
   - Recommend at least 8GB RAM
   - Can adjust `--max_steps` parameter to reduce memory usage

4. **KVM Unavailable**
   - KVM may not be available in cloud servers or virtual machines
   - Can run without KVM, but performance may be slower

### Logging and Debugging

Set log level:

```bash
python run.py --log_level DEBUG [other parameters]
```

View results:

```bash
# View result directory
ls -la ./results/

# View specific task trajectory
cat ./results/[domain]/[task_id]/traj.jsonl
```

## Performance Optimization

### Improving Execution Speed

1. Use KVM acceleration (if supported)
2. Adjust `--sleep_after_execution` parameter
3. Use more powerful hardware
4. Optimize Docker resource allocation

### Memory Optimization

1. Appropriately adjust `--max_steps`
2. Use smaller screen resolution
3. Limit Docker container memory usage

## Supported Models

The MGA system supports various foundation models:

### Commercial Models
- GPT-3.5 (gpt-3.5-turbo-16k, etc.)
- GPT-4 (gpt-4-0125-preview, gpt-4-1106-preview, etc.)
- GPT-4V (gpt-4-vision-preview, etc.)
- Gemini-Pro / Gemini-Pro-Vision
- Claude-3, Claude-2 series

### Open Source Models
- Mixtral 8x7B
- QWEN, QWEN-VL
- CogAgent
- Llama3
- GTA1 (recommended)

## Task Management and Results

### Unfinished Task Detection

The system automatically detects unfinished tasks and resumes from where it left off:

```python
# Check for unfinished tasks
test_file_list = get_unfinished(
    args.action_space,
    args.model,
    args.observation_type,
    args.result_dir,
    test_all_meta,
)
```

### Result Analysis

View current success rate:

```python
# Get current results
get_result(
    args.action_space,
    args.model,
    args.observation_type,
    args.result_dir,
    test_all_meta,
)
```

Results are stored in the following structure:
```
results/
├── pyautogui/
│   ├── screenshot/
│   │   ├── GTA1/
│   │   │   ├── domain1/
│   │   │   │   ├── task_id1/
│   │   │   │   │   ├── result.txt
│   │   │   │   │   ├── traj.jsonl
│   │   │   │   │   └── recording.mp4
```

## Advanced Configuration

### Multi-Environment Setup

For running multiple environments in parallel, use:

```bash
python run_multienv_gta1.py \
  --provider_name docker \
  --path_to_vm /path/to/Ubuntu.qcow2 \
  --headless \
  --observation_type screenshot \
  --model MGA1 \
  --num_envs 4 \
  --result_dir ./results
```

### Custom Domain Testing

To test specific domains:

```bash
python run.py \
  --domain "os,office" \
  [other parameters]
```

## More Information
If you want to know more detailed information for Osworld Implement, you can refer to their github(https://github.com/xlang-ai/OSWorld)



For issues, please check the project's Issue page or submit a new Issue.

## Citation

please cite:
```
