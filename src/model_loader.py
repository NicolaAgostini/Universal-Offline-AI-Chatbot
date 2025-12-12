from src.ollama_client import ask_ollama
from langchain.llms.base import LLM
from typing import Optional, List, Mapping
from src.config import MODEL
from langchain_core.pydantic_v1 import PrivateAttr

class OllamaLLM(LLM):
    _model_name: str = PrivateAttr()
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        print("[DEBUG] LLM LOADED WITH MODEL:", self._model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return ask_ollama(prompt, model_name=self._model_name)

    @property
    def _identifying_params(self) -> Mapping:
        return {"model": self._model_name}

    @property
    def _llm_type(self) -> str:
        # Serve a LangChain per identificare il tipo di LLM
        return "ollama"


def load_llm():
    """
    Restituisce un LLM compatibile con LangChain che usa Ollama moderno.
    """
    return OllamaLLM(MODEL)
