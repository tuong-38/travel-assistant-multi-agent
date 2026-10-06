from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def get_chat_model(model_alias: str) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model_alias,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key.get_secret_value(),
        max_retries=0,
    )
