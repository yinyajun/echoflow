from dataclasses import dataclass

from echoflow.llm.base_params import Params


@dataclass
class AnthropicParams(Params):
    top_k: int = 50
    model_id: str = "claude-3-5-sonnet-latest"
