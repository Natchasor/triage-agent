"""Tool abstraction and registry.

A tool is a name, a description, a Pydantic model for its arguments, and a run() method.
The JSON schema sent to the LLM and the validation of the LLM's arguments both come from
the same argument model, so the two can never drift apart.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, args: Any) -> dict[str, Any]:
        """Execute with validated arguments and return JSON-serialisable data."""

    def spec(self) -> dict[str, Any]:
        """The tool definition in OpenAI Responses API format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
            "strict": False,  # the Responses API defaults to strict, which needs every field required
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        """Run a tool requested by the LLM.

        Never raises. Every failure is returned as {"error": ...} so the model can read
        what went wrong and recover, instead of crashing the whole agent loop.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {self.names}"}

        try:
            args = tool.args_model.model_validate_json(raw_arguments or "{}")
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            return {"error": f"Invalid arguments for '{name}': {problems}"}

        try:
            return tool.run(args)
        except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the agent
            return {"error": f"Tool '{name}' failed: {exc}"}
