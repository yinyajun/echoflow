from echoflow.llm.base_tools import ToolWrapper


class AnthropicTool(ToolWrapper):
    def marshal(self):
        tool = self.clone()
        spec = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema.json_schema(extra={"key_any_of": "oneOf"}),
        }

        return spec
