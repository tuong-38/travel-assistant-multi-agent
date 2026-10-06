from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.state import ChatState


async def chat_node(state: ChatState) -> dict[str, list[dict[str, str]]]:
    latest_message = state["messages"][-1]["content"]
    return {"messages": [{"role": "assistant", "content": f"You said: {latest_message}"}]}


def build_graph(checkpointer: Any) -> Any:
    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder.compile(checkpointer=checkpointer)
