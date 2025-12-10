from src.ollama_client import ask_ollama
from langchain.llms.base import LLM
from typing import Optional, List, Mapping


class OllamaLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return ask_ollama(prompt)

    @property
    def _identifying_params(self) -> Mapping:
        return {"model": "mistral:instruct"}

    @property
    def _llm_type(self) -> str:
        # Serve a LangChain per identificare il tipo di LLM
        return "ollama"


def load_llm():
    """
    Restituisce un LLM compatibile con LangChain che usa Ollama moderno.
    """
    return OllamaLLM()
