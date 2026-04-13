"""
LLM Server Module

Provides unified interfaces for LLM (Large Language Model) interactions.

Usage:
    from mm_agents.llm_server import OpenAIClient
    
    # Create client
    client = OpenAIClient(model="gpt-4o")
    
    # Build messages with images
    messages = client.build_messages(
        system_prompt="You are helpful.",
        user_prompt="Describe this image.",
        images=[screenshot],
    )
    
    # Call API
    responses = client.call(messages)
"""

from .client import (
    LLMClient,
    OpenAIClient,
    LocalModelClient,
)

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "LocalModelClient",
]
