import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv, set_key

ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env",
)
load_dotenv(ENV_PATH)


class AIProviderSettings:
    def __init__(self):
        self.last_error = ""

    @property
    def provider(self) -> str:
        return os.getenv("AI_PROVIDER", "gemini").strip().lower()

    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")

    @property
    def ollama_model(self) -> str:
        return os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()

    def configure(self, provider: str, ollama_base_url: str, ollama_model: str) -> bool:
        provider = provider.strip().lower()
        base_url = ollama_base_url.strip().rstrip("/")
        model = ollama_model.strip()
        parsed_url = urlparse(base_url)
        if provider not in {"gemini", "ollama"}:
            raise ValueError("Provider must be 'gemini' or 'ollama'.")
        if (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc
                or parsed_url.username or parsed_url.password
                or parsed_url.path not in {"", "/"} or parsed_url.query or parsed_url.fragment):
            raise ValueError("Ollama URL must be an HTTP(S) server URL without credentials or a path.")
        if not model or len(model) > 128:
            raise ValueError("Ollama model must contain 1 to 128 characters.")

        try:
            if not os.path.exists(ENV_PATH):
                with open(ENV_PATH, "w", encoding="utf-8"):
                    pass
            set_key(ENV_PATH, "AI_PROVIDER", provider)
            set_key(ENV_PATH, "OLLAMA_BASE_URL", base_url)
            set_key(ENV_PATH, "OLLAMA_MODEL", model)
        except OSError as exc:
            self.last_error = f"Could not save AI provider settings ({type(exc).__name__})."
            return False

        os.environ["AI_PROVIDER"] = provider
        os.environ["OLLAMA_BASE_URL"] = base_url
        os.environ["OLLAMA_MODEL"] = model
        self.last_error = ""
        return True


class OllamaService:
    def __init__(self, settings: AIProviderSettings):
        self.settings = settings
        self.last_error = ""

    def status(self) -> dict:
        model = self.settings.ollama_model
        try:
            response = requests.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=2.0,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
            installed_models = {
                item.get("name") for item in models
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            return {
                "available": True,
                "model": model,
                "model_available": model in installed_models,
                "error": None if model in installed_models else f"Model '{model}' is not downloaded.",
            }
        except requests.RequestException as exc:
            return {
                "available": False,
                "model": model,
                "model_available": False,
                "error": f"Ollama is unreachable ({type(exc).__name__}).",
            }
        except (ValueError, AttributeError) as exc:
            return {
                "available": False,
                "model": model,
                "model_available": False,
                "error": f"Ollama returned an invalid status response ({type(exc).__name__}).",
            }

    def chat(self, messages: list[dict], response_format: str | None = None,
             timeout: float = 60.0) -> str | None:
        self.last_error = ""
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }
        if response_format:
            payload["format"] = response_format

        try:
            response = requests.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            if not isinstance(content, str) or not content.strip():
                self.last_error = "Ollama returned an empty response."
                return None
            return content.strip()
        except requests.Timeout:
            self.last_error = f"Ollama model {self.settings.ollama_model} request timed out."
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            self.last_error = f"Ollama HTTP {status_code}; check that the model is installed."
        except requests.RequestException as exc:
            self.last_error = f"Ollama request failed ({type(exc).__name__})."
        except (ValueError, AttributeError) as exc:
            self.last_error = f"Ollama returned an invalid response ({type(exc).__name__})."
        return None


ai_provider_settings = AIProviderSettings()
ollama_service = OllamaService(ai_provider_settings)
