"""Helpers for turning Pydantic models into JSON schemas the OpenAI API accepts."""
from typing import Any


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a Pydantic JSON schema with every $ref replaced by its definition.

    Pydantic puts enums and nested models under "$defs" and points at them with "$ref".
    Inlining them gives the API one flat, self-contained schema.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
            extras = {k: resolve(v) for k, v in node.items() if k != "$ref"}
            return {**target, **extras}
        return {k: resolve(v) for k, v in node.items() if k != "$defs"}

    return resolve(schema)
