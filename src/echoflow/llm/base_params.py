from dataclasses import dataclass


@dataclass
class Params:
    model_id: str
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.8
