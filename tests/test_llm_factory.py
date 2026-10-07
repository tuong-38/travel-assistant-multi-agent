from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.llm import factory


@pytest.mark.parametrize("model_alias", ["travel_general", "travel_local"])
def test_model_factory_uses_gateway_and_disables_retries(monkeypatch, model_alias: str) -> None:
    captured: dict = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: SimpleNamespace(
            litellm_base_url="http://litellm:4000/v1",
            litellm_api_key=SecretStr("local-test-key"),
        ),
    )

    model = factory.get_chat_model(model_alias)

    assert model is not None
    assert captured == {
        "model": model_alias,
        "base_url": "http://litellm:4000/v1",
        "api_key": "local-test-key",
        "max_retries": 0,
    }
