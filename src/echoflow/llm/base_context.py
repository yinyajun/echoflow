from dataclasses import dataclass, field

from echoflow.llm.base_messages import Messages
from echoflow.llm.base_params import Params
from echoflow.llm.base_tools import Tool


@dataclass
class CacheStrategy:
    cache_system: bool = False
    cache_history: bool = False
    cache_tool: bool = False


@dataclass
class LLMContext:
    params: Params = None
    system: Messages = None  # 路由问题？
    history: Messages = None
    rag: Messages = None  # 路由问题？
    tools: list[Tool] = field(default_factory=list)
