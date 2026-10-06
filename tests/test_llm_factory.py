from types import SimpleNamespace

from pydantic import SecretStr

from app.llm import factory


def test_model_factory_uses_gateway_and_disables_retries(monkeypatch) -> None:
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

    model = factory.get_chat_model("travel_general")

    assert model is not None
    assert captured == {
        "model": "travel_general",
        "base_url": "http://litellm:4000/v1",
        "api_key": "local-test-key",
        "max_retries": 0,
    }
