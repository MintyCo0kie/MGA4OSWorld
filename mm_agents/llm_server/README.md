# LLM Server Module

统一的 LLM (Large Language Model) 和 LMM (Large Multi-modal Model) 调用接口。

## 模块结构

```
llm_server/
├── __init__.py      # 模块入口，导出所有组件
├── client.py        # 底层 API 客户端
├── engine.py        # 中层引擎，处理消息和图像
├── agent.py         # 高层 Agent，提供便捷接口
└── README.md        # 本文档
```

## 组件说明

### 1. Client (客户端层)

底层 API 调用封装，处理网络请求和错误重试。

```python
from mm_agents.llm_server import OpenAIClient, LocalModelClient, GroundingClient

# OpenAI 兼容 API
client = OpenAIClient(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1/chat/completions",
)
responses = client.call(messages, model="gpt-4o")

# 本地模型 API
local_client = LocalModelClient(base_url="http://localhost:8000/v1/chat/completions")
responses = local_client.call(messages, model="llama3")

# Grounding 模型 API
grounding_client = GroundingClient(grounding_url="http://localhost:8001/ground")
result = grounding_client.ground(prompt="Click the submit button", image_base64="...")
```

### 2. Engine (引擎层)

处理消息构建、图像编码等中间逻辑。

```python
from mm_agents.llm_server import LMMEngineOpenAI, GroundingEngine

# OpenAI 引擎
engine = LMMEngineOpenAI(
    model="gpt-4o",
    api_key="your-api-key",
    max_tokens=4096,
)

# 纯文本生成
responses = engine.generate(messages)

# 带图像生成
responses = engine.generate_with_images(
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe this image.",
    images=[pil_image],
)

# Grounding 引擎
grounding_engine = GroundingEngine(grounding_url="http://localhost:8001/ground")
result = grounding_engine.ground_element("Submit button", screenshot)
```

### 3. Agent (代理层)

高层接口，提供对话管理和常用操作。

```python
from mm_agents.llm_server import LMMAgent, GroundingAgent

# 通用对话 Agent
agent = LMMAgent(model="gpt-4o")
agent.set_system_prompt("You are a GUI automation assistant.")

# 对话模式（保持历史）
response = agent.chat("What do you see on the screen?", images=[screenshot])
response = agent.chat("Now click the button.")

# 单次问答模式（不保持历史）
response = agent.ask("Describe this image.", images=[screenshot])

# 图像分析
description = agent.analyze_image(screenshot, prompt="List all buttons on screen.")

# Grounding Agent
grounding_agent = GroundingAgent(grounding_url="http://localhost:8001/ground")
location = grounding_agent.locate("Submit button", screenshot)
coords = grounding_agent.extract_coordinates(location, (1920, 1080))
```

## 使用工厂创建实例

```python
from mm_agents.llm_server import LMMEngineFactory, LLMClientFactory

# 根据模型名自动选择引擎
engine = LMMEngineFactory.create(
    model="gpt-4o",
    api_key="your-api-key",
    max_tokens=4096,
)

# 根据模型名自动选择客户端
client = LLMClientFactory.create(
    model="gpt-4o",
    api_key="your-api-key",
)
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_BASE_URL` | OpenAI API 端点 | `https://api.openai.com/v1/chat/completions` |
| `LOCAL_MODEL_URL` | 本地模型 API 端点 | - |
| `GROUNDING_URL` | Grounding 模型端点 | - |

## 与原有代码的兼容性

原有的 `LMMEngineOpenAI` 和 `LMMAgent` 类已被包装以保持向后兼容：

```python
# 原有用法仍然有效
from mm_agents.MGA_Agent import LMMEngineOpenAI, LMMAgent

engine = LMMEngineOpenAI(model="gpt-4o", api_key="...")
agent = LMMAgent(engine_params={...})
```

## 扩展新的 LLM 提供商

实现 `LLMClient` 接口：

```python
from mm_agents.llm_server import LLMClient

class MyCustomClient(LLMClient):
    def call(self, messages, model, max_tokens=4096, temperature=0.0, n=1, **kwargs):
        # 实现你的 API 调用逻辑
        pass
    
    def is_available(self):
        return True
```
