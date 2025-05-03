import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal

from echoflow.impl.anthropic.messages import AnthropicDynamicMessages, AnthropicStaticMessages
from echoflow.impl.anthropic.params import AnthropicParams
from echoflow.impl.anthropic.tools import AnthropicTool
from echoflow.llm.base_client import Client, StreamEvent, StreamEventType
from echoflow.llm.base_context import CacheStrategy, LLMContext
from echoflow.llm.base_messages import Messages, ToolCall
from echoflow.llm.base_params import Params
from echoflow.logger import get_logger

logger = get_logger()

try:
    import anthropic
    from anthropic.types import CacheControlEphemeralParam

except ModuleNotFoundError as e:
    logger.log_error(f"Exception: {e}")
    logger.log_error("In order to use Anthropic, you need to `pip install echoflow[anthropic]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AnthropicContext(LLMContext):
    params: AnthropicParams = AnthropicParams()
    system: AnthropicStaticMessages = field(
        default_factory=AnthropicStaticMessages
    )  # todo 路由问题？
    history: AnthropicStaticMessages = field(default_factory=AnthropicStaticMessages)
    rag: AnthropicDynamicMessages = field(
        default_factory=AnthropicStaticMessages
    )  # todo 路由问题？
    tools: list[AnthropicTool] = field(default_factory=list)


class AnthropicClient(Client):
    def __init__(
        self,
        provider: Literal["anthropic", "bedrock", "vertex"] = "anthropic",
        api_key: str = None,  # anthropic
        aws_region: str = None,  # bedrock
        aws_secret_key: str = None,  # bedrock
        aws_access_key: str = None,  # bedrock
        cache_strategy: CacheStrategy = CacheStrategy(),
    ):
        if provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

        elif provider == "bedrock":
            self.client = anthropic.AsyncAnthropicBedrock(
                aws_region=aws_region, aws_secret_key=aws_secret_key, aws_access_key=aws_access_key
            )

        elif provider == "vertex":
            self.client = anthropic.AsyncAnthropicVertex()

        else:
            raise ValueError("unsupported provider for AnthropicClient")

        self.cache_strategy = cache_strategy

    async def stream_generate(
        self, ctx: AnthropicContext, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        params = ctx.params
        tools = [t.marshal() for t in ctx.tools]
        system = ctx.system.value if ctx.system else []
        history = ctx.history.value if ctx.history else []

        if len(system) and self.cache_strategy.cache_system:
            system[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")

        if len(history) and self.cache_strategy.cache_history:
            history[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")

        # todo: dynamic

        stream = await self.client.messages.create(
            tools=tools,
            messages=history,
            model=params.model_id,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            max_tokens=params.max_tokens,
            system=system,
            stream=True,
        )
        async for event in self._process_stream(stream):
            yield event

    async def _process_stream(self, stream) -> AsyncGenerator[StreamEvent, None]:
        tool_id = None
        tool_name = None
        tool_arguments = ""
        meta = {}

        async for event in stream:
            logger.log_debug(f"A#### {event}")

            if event.type == "message_start":
                message = event.message
                usage = message.usage
                meta["input_usage"] = usage
                yield StreamEvent(type=StreamEventType.start, data={})
                continue

            elif event.type == "content_block_start":
                content_block = event.content_block
                if content_block.type == "tool_use":
                    tool_id = content_block.id
                    tool_name = content_block.name

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text_delta = delta.text
                    if not text_delta:
                        continue

                    yield StreamEvent(
                        type=StreamEventType.text_delta, data={"text_delta": text_delta}
                    )

                elif delta.type == "input_json_delta":
                    tool_arguments += delta.partial_json

            elif event.type == "content_block_stop":
                if tool_id:
                    try:
                        arguments = json.loads(tool_arguments)
                    except:
                        arguments = {}

                    tool_call_info = ToolCall(id=tool_id, name=tool_name, input=arguments)
                    yield StreamEvent(type=StreamEventType.tool, data={"tool": tool_call_info})

                    tool_id = None
                    tool_name = None
                    tool_arguments = ""

            elif event.type == "message_delta":
                stop_reason = event.delta.stop_reason
                yield StreamEvent(type=StreamEventType.stop, data={"stop_reason": stop_reason})

                usage = event.usage
                meta["output_usage"] = usage  # TODO: CHECK
                yield StreamEvent(type=StreamEventType.metadata, data={"metadata": meta})
