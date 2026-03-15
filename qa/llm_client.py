"""
qa/llm_client.py
LLM adapter for answer generation supporting:
  - GeminiAdapter   (default, uses google-genai)
  - OpenAIAdapter   (toggled via USE_OPENAI_API=true)
  - LocalLLMAdapter (Mistral/Llama via llama-cpp-python)

Public API:
  generate(prompt: str) -> str
  async_generate(prompt: str) -> str      (for batch parallel calls)
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from config import (
    GEMINI_API_KEY, OPENAI_API_KEY, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_CONCURRENCY,
    USE_OPENAI_API,
)

logger = logging.getLogger(__name__)


# ── Gemini Adapter ────────────────────────────────────────────────────────────

class GeminiAdapter:
    def __init__(self, model: str = LLM_MODEL):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model  = model
        self._types  = types

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[self._types.Content(
                role="user",
                parts=[self._types.Part.from_text(text=user_prompt)]
            )],
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            )
        )
        return resp.text or ""

    async def async_generate(self, system_prompt: str, user_prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, system_prompt, user_prompt)


# ── OpenAI Adapter ────────────────────────────────────────────────────────────

class OpenAIAdapter:
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI, AsyncOpenAI
        self._client  = OpenAI(api_key=OPENAI_API_KEY)
        self._aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._model   = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    async def async_generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = await self._aclient.chat.completions.create(
            model=self._model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""


# ── Local LLM Adapter (Mistral / Llama) ──────────────────────────────────────

class LocalLLMAdapter:
    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama
            self._llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        resp = self._llm(full_prompt, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
        return resp["choices"][0]["text"]

    async def async_generate(self, system_prompt: str, user_prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, system_prompt, user_prompt)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_adapter():
    """Return the correct LLM adapter based on config flags."""
    if USE_OPENAI_API and OPENAI_API_KEY:
        logger.info("LLM: Using OpenAI adapter (%s)", LLM_MODEL)
        return OpenAIAdapter(model=LLM_MODEL)
    if GEMINI_API_KEY:
        logger.info("LLM: Using Gemini adapter (%s)", LLM_MODEL)
        return GeminiAdapter(model=LLM_MODEL)
    raise ValueError(
        "No LLM adapter available. Set GEMINI_API_KEY or OPENAI_API_KEY in .env"
    )


# ── Batch parallel execution ──────────────────────────────────────────────────

async def batch_generate(prompts: List[tuple[str, str]], concurrency: int = LLM_CONCURRENCY) -> List[str]:
    """
    Run multiple (system_prompt, user_prompt) pairs concurrently.
    Returns list of answers in the same order.
    """
    adapter = get_adapter()
    sem     = asyncio.Semaphore(concurrency)

    async def _call(sys_p: str, usr_p: str) -> str:
        async with sem:
            return await adapter.async_generate(sys_p, usr_p)

    tasks  = [_call(sp, up) for sp, up in prompts]
    return await asyncio.gather(*tasks, return_exceptions=False)
