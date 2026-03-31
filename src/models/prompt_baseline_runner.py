from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.data.schema import Prediction


def extract_prediction(raw_response: str) -> tuple[str, str]:
    raw_response = raw_response.strip()
    try:
        parsed = json.loads(raw_response)
        answer = parsed.get("answer", "").strip().lower()
        confidence = parsed.get("confidence", "unknown")
        return answer, confidence
    except json.JSONDecodeError:
        return raw_response.split()[0].strip().lower(), "unknown"


def _normalize_binary_answer(answer: str) -> str:
    answer = answer.strip().lower()
    if answer in {"yes", "no"}:
        return answer
    if answer.startswith("yes"):
        return "yes"
    if answer.startswith("no"):
        return "no"
    return answer


def _to_file_uri(path_str: str, project_root: Path) -> str:
    if path_str.startswith(("file://", "http://", "https://")):
        return path_str
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return f"file://{path}"


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    return {"raw_response_repr": repr(response)}


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    output = response_payload.get("output")
    if not output:
        raise RuntimeError(
            "DashScope returned no output. "
            f"Full response: {json.dumps(response_payload, ensure_ascii=False, default=str)}"
        )

    choices = output.get("choices") or []
    if not choices:
        raise RuntimeError(
            "DashScope returned output without choices. "
            f"Full response: {json.dumps(response_payload, ensure_ascii=False, default=str)}"
        )

    message = choices[0].get("message") or {}
    content = message.get("content") or []
    if not content:
        raise RuntimeError(
            "DashScope returned a choice without message content. "
            f"Full response: {json.dumps(response_payload, ensure_ascii=False, default=str)}"
        )

    first_item = content[0]
    if isinstance(first_item, dict) and "text" in first_item:
        return first_item["text"]
    if isinstance(first_item, str):
        return first_item

    raise RuntimeError(
        "DashScope returned an unsupported message content shape. "
        f"Full response: {json.dumps(response_payload, ensure_ascii=False, default=str)}"
    )


class DashScopeProcedureRunner:
    """DashScope SDK runner using local file paths for multimodal inputs."""

    backend_name = "dashscope_api"

    def __init__(
        self,
        model_id: str,
        fps: int = 2,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
        project_root: str | Path = ".",
    ) -> None:
        self.model_id = model_id
        self.fps = fps
        self.api_key_env = api_key_env
        self.project_root = Path(project_root).resolve()

        try:
            import dashscope  # type: ignore
        except ImportError as exc:
            raise RuntimeError("dashscope is not installed in the current environment") from exc

        self.dashscope = dashscope
        if base_url:
            self.dashscope.base_http_api_url = base_url

        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")

    def _convert_content_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_type = item.get("type")
        if item_type == "text":
            return {"text": item["text"]}
        if item_type == "image":
            return {"image": _to_file_uri(item["image"], self.project_root)}
        if item_type == "video":
            return {
                "video": _to_file_uri(item["video"], self.project_root),
                "fps": self.fps,
            }
        raise ValueError(f"Unsupported content item type: {item_type}")

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            converted.append(
                {
                    "role": message["role"],
                    "content": [self._convert_content_item(item) for item in message["content"]],
                }
            )
        return converted

    def infer(self, prepared_input: dict[str, Any]) -> Prediction:
        start = time.perf_counter()
        messages = self._convert_messages(prepared_input["messages"])
        completion = self.dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_id,
            messages=messages,
            result_format="message",
        )

        response_payload = _response_to_dict(completion)
        raw_response = _extract_response_text(response_payload)
        predicted_label, confidence = extract_prediction(raw_response)
        predicted_label = _normalize_binary_answer(predicted_label)
        latency_ms = int((time.perf_counter() - start) * 1000)

        return Prediction(
            trial_id=prepared_input["trial_id"],
            query_video_path=prepared_input["query_video_path"],
            gold_label=prepared_input["gold_label"],
            predicted_label=predicted_label,
            raw_response=raw_response,
            confidence=confidence,
            backend=self.backend_name,
            latency_ms=latency_ms,
        )
