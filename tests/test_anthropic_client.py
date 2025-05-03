import os
import unittest

from echoflow.impl.anthropic.client import AnthropicClient, AnthropicContext
from echoflow.impl.anthropic.messages import (
    AnthropicDynamicMessages,
    AnthropicStaticMessages,
)
from echoflow.llm.base_messages import Message, ToolCall, ToolResult


class TestAnthropicClientWithRaw(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = AnthropicClient(api_key=os.getenv("anthropic_api_key"))
        self.ctx = AnthropicContext()

    async def test_stream_generate(self):
        async for event in self.client.stream_generate(self.ctx):
            print(777, event)
