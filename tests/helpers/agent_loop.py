from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any


def make_response(
    content: Any = None,
    *,
    response_id: str = "resp-1",
    choices: list[Any] | None = None,
) -> Any:
    if choices is None:
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        choices = [choice]
    return SimpleNamespace(id=response_id, choices=choices)


def make_response_without_message(*, response_id: str = "resp-1") -> Any:
    return make_response(
        response_id=response_id,
        choices=[SimpleNamespace()],
    )


def sequence_call_llm(responses: Iterable[Any]) -> Callable[..., Any]:
    iterator = iter(responses)

    def fake_call_llm(**_: Any) -> Any:
        return next(iterator)

    return fake_call_llm


def install_warnings(monkeypatch: Any, target: str) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(target, lambda message: warnings.append(message))
    return warnings
