from dataclasses import dataclass, field

from echoflow.llm.base_messages import Messages
from echoflow.llm.base_tools import Tool


@dataclass
class CacheStrategy:
    cache_system: bool = False
    cache_history: bool = False
    cache_tool: bool = False


@dataclass
class Params:
    model_id: str
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.8


@dataclass
class LLMContext:
    params: Params = None
    system: Messages = None  # 路由问题？交给上层
    history: Messages = None
    rag: Messages = None  # 路由问题？交给上层
    tools: list[Tool] = field(default_factory=list)  # 路由问题？交给上层
