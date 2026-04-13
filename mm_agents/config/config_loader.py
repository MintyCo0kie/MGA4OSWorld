"""
Configuration loader for MGA Agent.
Supports YAML config files with environment variable fallback.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import yaml

logger = logging.getLogger("desktopenv.config")


@dataclass
class CommonConfig:
    api_key: str = ""
    platform: str = "ubuntu"
    max_tokens: int = 8192
    temperature: float = 0.9
    top_p: float = 0.9
    max_steps: int = 30
    max_image_history_length: int = 5
    N_SEQ: int = 1
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    width: int = 1920
    height: int = 1080
    #=== kimi ===
    screen_size: Tuple[int, int] = (1920, 1080) # The screen size
    coordinate_type: str = "relative" # The coordinate type: relative, absolute, qwen25
    thinking: bool = True


@dataclass
class PlannerConfig:
    model: str = "gpt-5"
    client_type: str = "openai"
    base_url: str = ""
    api_key: str = ""



@dataclass
class MemoryConfig:
    enabled: bool = True
    model: str = "qwen"
    client_type: str = "local"
    base_url: str = ""
    api_key: str = ""


@dataclass
class ObserverConfig:
    enabled: bool = False
    model: str = "observer"
    client_type: str = "local"
    base_url: str = ""
    api_key: str = ""


@dataclass
class GroundingConfig:
    model: str = "grounding"
    client_type: str = "local"
    base_url: str = ""
    api_key: str = ""


@dataclass
class MGA_Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)

    def resolve_api_keys(self):
        """
        解析 API Key 优先级：
        模块级 api_key > common.api_key > 环境变量 OPENAI_API_KEY
        """
        # common.api_key 的 fallback
        if not self.common.api_key:
            self.common.api_key = os.environ.get("OPENAI_API_KEY", "")

        # 各模块：如果自己没配 api_key，则使用 common.api_key
        for module in [self.planner, self.memory, self.observer, self.grounding]:
            if not module.api_key:
                module.api_key = self.common.api_key

    def resolve_base_urls(self):
        """
        解析 base_url 的环境变量 fallback
        """
        env_mapping = {
            "planner": "PLANNER_URL",
            "memory": "MEMORY_URL",
            "observer": "OBSERVER_URL",
            "grounding": "GROUNDING_URL",
        }
        for name, env_var in env_mapping.items():
            module = getattr(self, name)
            if not module.base_url:
                module.base_url = os.environ.get(env_var, "")


def load_config(config_path: Optional[str] = None) -> MGA_Config:
    """
    加载配置文件。

    优先级：
    1. 指定的 config_path
    2. 环境变量 MGA_CONFIG_PATH
    3. 默认路径 mm_agents/config/config.yaml

    Args:
        config_path: YAML 配置文件路径

    Returns:
        MGA_Config 实例
    """
    if config_path is None:
        config_path = os.environ.get(
            "MGA_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "config.yaml"),
        )

    config = MGA_Config()

    if os.path.exists(config_path):
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # 填充各模块
        if "common" in raw:
            config.common = CommonConfig(**{
                k: v for k, v in raw["common"].items()
                if k in CommonConfig.__dataclass_fields__
            })


        if "planner" in raw:
            config.planner = PlannerConfig(**{
                k: v for k, v in raw["planner"].items()
                if k in PlannerConfig.__dataclass_fields__
            })

        if "memory" in raw:
            config.memory = MemoryConfig(**{
                k: v for k, v in raw["memory"].items()
                if k in MemoryConfig.__dataclass_fields__
            })

        if "observer" in raw:
            config.observer = ObserverConfig(**{
                k: v for k, v in raw["observer"].items()
                if k in ObserverConfig.__dataclass_fields__
            })

        if "grounding" in raw:
            config.grounding = GroundingConfig(**{
                k: v for k, v in raw["grounding"].items()
                if k in GroundingConfig.__dataclass_fields__
            })
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # 解析 API Key 和 URL 的 fallback
    config.resolve_api_keys()
    config.resolve_base_urls()

    logger.info(f"Config loaded: planner={config.planner.client_type}/{config.planner.model}, "
                f"memory={'enabled' if config.memory.enabled else 'disabled'}, "
                f"observer={'enabled' if config.observer.enabled else 'disabled'}")

    return config