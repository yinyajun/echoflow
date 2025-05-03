from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Union

Text = str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    id: str
    content: str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: List[Union[Text, ToolCall, ToolResult]]


class MessageAdapter(ABC):
    @abstractmethod
    def adapt(self, message: Message) -> Any:
        pass

    @abstractmethod
    def to_message(self, element) -> Message:
        pass


class Messages(ABC, list):
    @abstractmethod
    def add_message(self, message: Message):
        pass


class _MergedMessages(Messages):
    def _alternate_role(self, role: Literal["user", "assistant", "tool"]) -> bool:
        if len(self) == 0 or self[-1].role != role:
            self.append(Message(role=role, content=[]))
            return True
        return False

    def _add_text(self, role: Literal["user", "assistant"], text: str):
        if not text:
            return

        self._alternate_role(role)
        self[-1].content.append(text)

    def _add_tool_call(self, tool_call: ToolCall):
        if not tool_call:
            return

        self._alternate_role("assistant")
        self[-1].content.append(tool_call)

    def _add_tool_result(self, tool_result: ToolResult):
        if not tool_result:
            return

        self._alternate_role("tool")
        self[-1].content.append(tool_result)

    def add_message(self, message: Message):
        match message.role:
            case "user" | "assistant":
                for i, content in enumerate(message.content):
                    if isinstance(content, Text):
                        self._add_text(message.role, content)
                    elif isinstance(content, ToolCall):
                        self._add_tool_call(content)
                    else:
                        raise TypeError(f"{type(content)} content is not supported")

            case "tool":
                tool_result = message.content[0]
                assert isinstance(tool_result, ToolResult), "Expected type is <ToolResult>"
                self._add_tool_result(tool_result)

            case _:
                raise ValueError(f"role {message.role} is not supported")


class StaticMessages(_MergedMessages):
    def __init__(self, adapter: MessageAdapter = None):
        super().__init__()
        self.adapter = adapter
        self._static = []

    def _alternate_role(self, role: Literal["user", "assistant", "tool"]) -> bool:
        alternated = super()._alternate_role(role)
        if alternated:
            self._static.append(Message(role=role, content=[]))
        return alternated

    def add_message(self, message: Message):
        super().add_message(message)
        self._static[-1] = self.adapter.adapt(self[-1]) if self.adapter else self[-1]

    @property
    def value(self):
        return self._static


class DynamicMessages(Messages):
    def __init__(self, adapter: MessageAdapter = None):
        super().__init__()
        self.adapter = adapter

    def add_message(self, message: Message):
        self.append(message)

    @property
    def value(self):
        res = StaticMessages(self.adapter)

        for i, m in enumerate(self):
            res.add_message(m)

        return res.value
