"""Chat backends. The reply logic only needs `generate(messages)` or `stream(messages)`."""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

Messages = list[dict[str, str]]


class ChatBackend(ABC):
    name = "backend"

    @abstractmethod
    def generate(self, messages: Messages, max_new_tokens: int = 200, temperature: float = 0.6) -> str: ...

    def stream(self, messages: Messages, max_new_tokens: int = 200, temperature: float = 0.6) -> Iterator[str]:
        """Yield the reply growing piece by piece (default: the whole reply at once)."""
        yield self.generate(messages, max_new_tokens, temperature)


class LlamaCppBackend(ChatBackend):
    """Quantised GGUF model on CPU through llama.cpp: no GPU, no API key, no quota."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: int | None = None,
    ):
        from llama_cpp import Llama

        self.name = f"{repo_id}/{filename}"
        self.llm = Llama.from_pretrained(
            repo_id=repo_id, filename=filename, n_ctx=n_ctx, n_threads=n_threads, verbose=False
        )

    def generate(self, messages, max_new_tokens=200, temperature=0.6):
        out = self.llm.create_chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=temperature)
        return out["choices"][0]["message"]["content"].strip()

    def stream(self, messages, max_new_tokens=200, temperature=0.6):
        text = ""
        for chunk in self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=temperature, stream=True
        ):
            text += chunk["choices"][0]["delta"].get("content") or ""
            yield text.strip()


class HFInferenceBackend(ChatBackend):
    """Hugging Face Inference Providers (free quota is small). The token comes from `HF_TOKEN`."""

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct", token: str | None = None):
        from huggingface_hub import InferenceClient

        self.name = f"hf-inference:{model}"
        self.client = InferenceClient(model=model, token=token or os.environ.get("HF_TOKEN"))

    def generate(self, messages, max_new_tokens=200, temperature=0.6):
        out = self.client.chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=temperature)
        return out.choices[0].message.content.strip()

    def stream(self, messages, max_new_tokens=200, temperature=0.6):
        text = ""
        for chunk in self.client.chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=temperature, stream=True
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                yield text.strip()


class TransformersBackend(ChatBackend):
    """The original setup: Llama 3 8B Instruct with `transformers` (gated model, needs a GPU or lots of RAM).

    Kept for the desktop app; not exercised by the tests or the Space.
    """

    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", token: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = token or os.environ.get("HF_TOKEN")  # None: falls back to `huggingface-cli login`
        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token
        )

    def generate(self, messages, max_new_tokens=200, temperature=0.6):
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        eos = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()


def create_backend(kind: str | None = None) -> ChatBackend:
    """`kind` (or the EMOTIONLLM_BACKEND variable): `llamacpp` (default), `hf-api` or `llama3-8b`."""
    kind = (kind or os.environ.get("EMOTIONLLM_BACKEND", "llamacpp")).lower()
    if kind == "llamacpp":
        return LlamaCppBackend(n_threads=int(os.environ.get("EMOTIONLLM_THREADS", "0")) or None)
    if kind == "hf-api":
        return HFInferenceBackend()
    if kind == "llama3-8b":
        return TransformersBackend()
    raise ValueError(f"unknown backend {kind!r}; use llamacpp, hf-api or llama3-8b")
