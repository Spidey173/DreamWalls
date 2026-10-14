import logging
from app.core.config import settings
from app.services.llm.base import BaseLLMClient
from app.services.llm.mock_client import MockLLMClient

logger = logging.getLogger("paintmywall.llm")

def get_llm_client() -> BaseLLMClient:
    """
    Factory function returning the configured LLM client.
    Falls back gracefully to MockLLMClient if no valid API key is present.
    """
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "gemini":
        key = settings.GEMINI_API_KEY.strip()
        if key and not key.startswith("your_"):
            try:
                from app.services.llm.gemini_client import GeminiClient
                return GeminiClient(api_key=key, model=settings.GEMINI_MODEL)
            except Exception as e:
                logger.warning(f"Failed to load GeminiClient: {e}; falling back to MockLLMClient.")
        logger.warning("GEMINI_API_KEY is unset or default placeholder; using MockLLMClient for development.")
        return MockLLMClient()

    elif provider == "anthropic":
        key = settings.ANTHROPIC_API_KEY.strip()
        if key and not key.startswith("your_"):
            try:
                from app.services.llm.anthropic_client import AnthropicClient
                return AnthropicClient(api_key=key, model=settings.ANTHROPIC_MODEL)
            except Exception as e:
                logger.warning(f"Failed to load AnthropicClient: {e}; falling back to MockLLMClient.")
        logger.warning("ANTHROPIC_API_KEY is unset or default placeholder; using MockLLMClient for development.")
        return MockLLMClient()

    elif provider == "groq":
        key = settings.GROQ_API_KEY.strip()
        if key and not key.startswith("your_"):
            try:
                from app.services.llm.groq_client import GroqClient
                return GroqClient(
                    api_key=key,
                    model=settings.GROQ_MODEL,
                    vision_model=settings.GROQ_VISION_MODEL
                )
            except Exception as e:
                logger.warning(f"Failed to load GroqClient: {e}; falling back to MockLLMClient.")
        logger.warning("GROQ_API_KEY is unset or default placeholder; using MockLLMClient for development.")
        return MockLLMClient()

    else:
        logger.warning(f"Unknown LLM_PROVIDER '{provider}'; defaulting to MockLLMClient.")
        return MockLLMClient()

