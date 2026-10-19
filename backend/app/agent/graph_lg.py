"""事件编排：用 LangGraph 把「事件类型 → 处理器」建成一张图。

为什么要有这一层
- 手写 `if event.type == ...` 的分发在分支变多后不可视化；图能把
  "有哪几种事件、各自去哪"画出来，加分支只需加边。
- 这一层**只做路由**，不搬运业务逻辑：节点内部仍调用原有的
  `_handle_user_message` / `_handle_finish_requested`，
  所以切换编排方式不会改变行为。

降级策略（很重要）
- langgraph 未安装或本层异常时，调用方（runner）会退回手写分发，
  **不会因为缺依赖导致整个服务起不来**。
- 默认编排器仍是手写；把 `AGENT_ORCHESTRATOR` 设为 `langgraph` 才启用本图。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypedDict

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 依赖是否存在由环境决定
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False
    logger.debug("langgraph unavailable: %s", exc)

# 事件类型
EVENT_USER_MESSAGE = "user_message"
EVENT_FINISH = "finish_requested"

# 图中节点名
NODE_CLASSIFY = "classify"
NODE_USER_MESSAGE = "user_message"
NODE_FINISH = "finish"
NODE_UNSUPPORTED = "unsupported"

Dispatch = Callable[[], Awaitable[Any]]


class EventGraphState(TypedDict, total=False):
    """图的输入输出。dispatch_* 是本次请求的闭包，随 state 传入。"""

    event_type: str
    kind: str
    result: Any
    dispatch_user: Dispatch
    dispatch_finish: Dispatch


def classify(event_type: str | None) -> str:
    """把事件类型映射到节点名。未识别的事件走 unsupported。"""
    if event_type == EVENT_USER_MESSAGE:
        return NODE_USER_MESSAGE
    if event_type == EVENT_FINISH:
        return NODE_FINISH
    return NODE_UNSUPPORTED


def build_event_graph():
    """构造并编译事件路由图。langgraph 不可用时抛 RuntimeError。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed")

    graph = StateGraph(EventGraphState)

    def _classify(state: EventGraphState) -> dict:
        return {"kind": classify(state.get("event_type"))}

    async def _user_message(state: EventGraphState) -> dict:
        return {"result": await state["dispatch_user"]()}

    async def _finish(state: EventGraphState) -> dict:
        return {"result": await state["dispatch_finish"]()}

    def _unsupported(state: EventGraphState) -> dict:
        return {"result": None}

    graph.add_node(NODE_CLASSIFY, _classify)
    graph.add_node(NODE_USER_MESSAGE, _user_message)
    graph.add_node(NODE_FINISH, _finish)
    graph.add_node(NODE_UNSUPPORTED, _unsupported)

    graph.set_entry_point(NODE_CLASSIFY)
    graph.add_conditional_edges(
        NODE_CLASSIFY,
        lambda state: state["kind"],
        {
            NODE_USER_MESSAGE: NODE_USER_MESSAGE,
            NODE_FINISH: NODE_FINISH,
            NODE_UNSUPPORTED: NODE_UNSUPPORTED,
        },
    )
    for node in (NODE_USER_MESSAGE, NODE_FINISH, NODE_UNSUPPORTED):
        graph.add_edge(node, END)

    return graph.compile()


_compiled = None


def get_event_graph():
    """进程内复用编译结果（编译有成本，不要每请求新建）。"""
    global _compiled
    if _compiled is None:
        _compiled = build_event_graph()
    return _compiled


async def run_event_via_graph(
    event_type: str | None,
    dispatch_user: Dispatch,
    dispatch_finish: Dispatch,
) -> Any:
    """按图路由并返回处理器结果。未识别事件返回 None。"""
    app = get_event_graph()
    out = await app.ainvoke({
        "event_type": event_type or "",
        "dispatch_user": dispatch_user,
        "dispatch_finish": dispatch_finish,
    })
    return out.get("result")
