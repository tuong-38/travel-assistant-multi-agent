import operator
from typing import Annotated, TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[dict[str, str]], operator.add]
