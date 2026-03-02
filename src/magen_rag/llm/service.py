from __future__ import annotations

import re

import httpx


class LLMService:
    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip() or "INSUFFICIENT_EVIDENCE"
        except Exception:
            return "INSUFFICIENT_EVIDENCE\n- Clarify target document or time range."


def extract_citations(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[[^\]]+:[^\]]+:[^\]]+\]", text)))
