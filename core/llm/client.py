# core/llm/client.py
import os


class LLMClient:
    """ Adds support for vllm API.
    Set LLM_BACKEND=vllm to use vllm, otherwise uses ollama.
    Set VLLM_BASE_URL if your vllm server isn't at localhost:8000.
    """

    def __init__(self, timeout=180.0):
        self.backend = os.getenv("LLM_BACKEND", "ollama")
        self.timeout = timeout

        # if vllm (cluster)
        if self.backend == "vllm":
            from openai import OpenAI
            self._client = OpenAI(
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                api_key="EMPTY",
                timeout=timeout,
            )
        # if ollama (laptop)
        else:
            import ollama
            self._client = ollama.Client(timeout=timeout)

    def chat(self, model, messages, format=None, options=None):
        options = options or {}

        if self.backend == "vllm":
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": options.get("temperature", 0.7),
            }
            if format is not None:
                kwargs["extra_body"] = {"guided_json": format}

            raw = self._client.chat.completions.create(**kwargs)
            return _OllamaShape(raw)
        else:
            return self._client.chat(
                model=model,
                messages=messages,
                format=format,
                options=options,
            )


class _OllamaShape:
    """openai to ollama"""

    def __init__(self, openai_resp):
        self.message = type("M", (), {"content": openai_resp.choices[0].message.content})()
        self.prompt_eval_count = openai_resp.usage.prompt_tokens if openai_resp.usage else 0
        self.eval_count = openai_resp.usage.completion_tokens if openai_resp.usage else 0
        self._raw = openai_resp

    def model_dump(self):
        return self._raw.model_dump()