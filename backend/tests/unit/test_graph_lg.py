"""事件编排图的单测。

langgraph 未安装时整组跳过——这正是降级路径存在的原因：
没有这个依赖，服务仍能用「手写分发」运行。
"""

from __future__ import annotations

import pytest

from app.agent import graph_lg
from app.agent.graph_lg import (
    EVENT_FINISH,
    EVENT_USER_MESSAGE,
    NODE_FINISH,
    NODE_UNSUPPORTED,
    NODE_USER_MESSAGE,
    classify,
)

requires_langgraph = pytest.mark.skipif(
    not graph_lg.LANGGRAPH_AVAILABLE, reason="langgraph not installed"
)


def test_classify_maps_event_types():
    assert classify(EVENT_USER_MESSAGE) == NODE_USER_MESSAGE
    assert classify(EVENT_FINISH) == NODE_FINISH
    assert classify("something_else") == NODE_UNSUPPORTED
    assert classify(None) == NODE_UNSUPPORTED


@requires_langgraph
def test_graph_routes_user_message():
    import asyncio

    calls = {"user": 0, "finish": 0}

    async def dispatch_user():
        calls["user"] += 1
        return "USER_RESULT"

    async def dispatch_finish():
        calls["finish"] += 1
        return "FINISH_RESULT"

    result = asyncio.run(
        graph_lg.run_event_via_graph("user_message", dispatch_user, dispatch_finish)
    )
    assert result == "USER_RESULT"
    assert calls == {"user": 1, "finish": 0}


@requires_langgraph
def test_graph_routes_finish():
    import asyncio

    async def dispatch_user():
        return "USER_RESULT"

    async def dispatch_finish():
        return "FINISH_RESULT"

    result = asyncio.run(
        graph_lg.run_event_via_graph("finish_requested", dispatch_user, dispatch_finish)
    )
    assert result == "FINISH_RESULT"


@requires_langgraph
def test_graph_unsupported_returns_none():
    import asyncio

    async def dispatch_user():
        raise AssertionError("must not be called")

    async def dispatch_finish():
        raise AssertionError("must not be called")

    result = asyncio.run(
        graph_lg.run_event_via_graph("unknown_event", dispatch_user, dispatch_finish)
    )
    assert result is None


@requires_langgraph
def test_graph_is_cached():
    graph_lg._compiled = None
    first = graph_lg.get_event_graph()
    second = graph_lg.get_event_graph()
    assert first is second
