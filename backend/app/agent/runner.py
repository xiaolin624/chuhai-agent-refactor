import logging
import time as _time
from collections.abc import AsyncGenerator

from app.agent.nodes import apply_extraction_result
from app.agent.state_machine import append_assistant_message, append_user_message
from app.agent.tools.executor import ToolExecutor
from app.agent.tools.external import register_external_tools
from app.agent.tools.external.dialogue import DialogueDeepSeekInput, _build_dialogue_input, _build_dialogue_prompt
from app.agent.tools.external.extraction import ExtractAnswersDeepSeekInput
from app.agent.mcp.adapter import register_enabled_mcp_tools
from app.agent.skills.bundled import register_all_bundled_skills
from app.agent.skills.registry import get_skill
from app.agent.tools.base import ToolContext, ToolErrorCode
from app.agent.tools.local import register_local_tools
from app.agent.tools.local.readiness import ReadinessCheckInput, ReadinessResult
from app.agent.tools.registry import ToolRegistry
from app.schemas.agent_protocol import AgentEvent, AgentRunResult, AgentTraceEvent, TerminalState
from app.schemas.agent_state import AgentState
from app.schemas.conversation import ConversationClientState
from app.schemas.extraction import ExtractionResult
from app.schemas.llm import LLMMessage
from app.services.deepseek_client import DeepSeekClient
from config import get_settings

logger = logging.getLogger(__name__)


def _default_max_agent_steps() -> int:
    return get_settings().MAX_AGENT_STEPS


async def _build_tool_registry(db_session=None) -> ToolRegistry:
    registry = ToolRegistry()
    register_local_tools(registry)
    register_external_tools(registry)
    if db_session is not None:
        await register_enabled_mcp_tools(registry, db_session)
    return registry


def _skill_name_from_message(message: str | None) -> str | None:
    if not message:
        return None
    text = message.strip()
    if not text.startswith("/"):
        return None
    name = text[1:].split(maxsplit=1)[0].strip()
    return name or None


def _parse_remember_command(message: str | None) -> tuple[str, str] | None:
    if not message:
        return None
    text = message.strip()
    if not text.startswith("/remember"):
        return None

    content = text[len("/remember"):].strip()
    if not content:
        return None

    for sep in ("：", ":"):
        if sep in content:
            name, body = content.split(sep, 1)
            name = name.strip()
            body = body.strip()
            if name and body:
                return name[:80], body
            break

    return content[:80], content


def _safe_error(msg: str) -> str:
    """安全文案：不泄露原始异常给用户。"""
    return "AI 暂时不可用，请稍后重试"


def _build_memory_query(state: AgentState, user_message: str) -> str:
    parts = [user_message]
    if state.slots.industry and state.slots.industry.value:
        parts.append(str(state.slots.industry.value))
    if state.slots.main_product and state.slots.main_product.value:
        parts.append(str(state.slots.main_product.value))
    return " ".join(parts)


async def _recall_memory_entries(
    executor: ToolExecutor,
    state: AgentState,
    user_message: str,
) -> list:
    query = _build_memory_query(state, user_message)
    from app.schemas.memory import MemoryRecallInput
    result = await executor.execute("memory.recall", MemoryRecallInput(query=query, limit=3))
    if result.error is None and result.data is not None:
        return getattr(result.data, "entries", [])
    return []


async def _recall_memory_entries_with_trace_status(
    executor: ToolExecutor,
    state: AgentState,
    user_message: str,
) -> tuple[list, bool]:
    """Recall memory entries and return (entries, failed).
    failed=True when memory.recall returns ToolResult.error.
    Programming exceptions still propagate.
    """
    query = _build_memory_query(state, user_message)
    from app.schemas.memory import MemoryRecallInput
    result = await executor.execute("memory.recall", MemoryRecallInput(query=query, limit=3))
    if result.error is not None:
        return [], True
    entries = getattr(result.data, "entries", []) if result.data is not None else []
    return entries, False


async def run_agent_event(
    state: AgentState,
    event: AgentEvent,
    registry: ToolRegistry | None = None,
    max_steps: int | None = None,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_model: str | None = None,
    db_session=None,
) -> AgentRunResult:
    effective_max_steps = max_steps if max_steps is not None else _default_max_agent_steps()
    if effective_max_steps <= 0:
        return AgentRunResult(
            state=state,
            terminal=TerminalState.MAX_STEPS_EXCEEDED,
        )

    async def _get_registry() -> ToolRegistry:
        nonlocal registry
        if registry is None:
            registry = await _build_tool_registry(db_session)
        return registry

    async def _dispatch_user_message() -> AgentRunResult:
        return await _handle_user_message(
            state, event, await _get_registry(),
            provider_base_url=provider_base_url,
            provider_api_key=provider_api_key,
            provider_model=provider_model,
            db_session=db_session,
        )

    async def _dispatch_finish() -> AgentRunResult:
        return await _handle_finish_requested(
            state, await _get_registry(),
            provider_base_url=provider_base_url,
            provider_api_key=provider_api_key,
            provider_model=provider_model,
            db_session=db_session,
        )

    # 编排器：默认手写分发；设为 langgraph 时走事件图（图内节点仍调用上面两个处理器，
    # 行为不变，只把"事件类型 → 处理器"变成可视化路由）。langgraph 缺失时自动降级。
    if get_settings().AGENT_ORCHESTRATOR == "langgraph":
        from app.agent.graph_lg import LANGGRAPH_AVAILABLE, run_event_via_graph

        if LANGGRAPH_AVAILABLE:
            result = await run_event_via_graph(
                event.type, _dispatch_user_message, _dispatch_finish
            )
            if result is not None:
                return result
            return AgentRunResult(state=state, terminal=TerminalState.FAILED)
        logger.warning(
            "AGENT_ORCHESTRATOR=langgraph 但 langgraph 未安装，退回手写分发"
        )

    if event.type == "user_message":
        return await _dispatch_user_message()

    if event.type == "finish_requested":
        return await _dispatch_finish()

    return AgentRunResult(state=state, terminal=TerminalState.FAILED)


async def _handle_user_message(
    state: AgentState,
    event: AgentEvent,
    registry: ToolRegistry,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_model: str | None = None,
    db_session=None,
) -> AgentRunResult:
    executor = ToolExecutor(registry)
    ctx = ToolContext(
        db_session=db_session,
        provider_base_url=provider_base_url,
        provider_api_key=provider_api_key,
        provider_model=provider_model,
    )
    current = append_user_message(state, event.message)

    remember = _parse_remember_command(event.message)
    if remember is not None:
        from app.schemas.memory import MemorySaveInput
        name, content = remember
        save_result = await executor.execute(
            "memory.save",
            MemorySaveInput(
                name=name,
                description=f"用户手动保存的长期记忆：{name}",
                type="user",
                content=content,
            ),
            context=ctx,
        )
        if save_result.error is not None:
            assistant_text = "记忆保存失败，请检查内容后重试"
        else:
            assistant_text = f"已保存到长期记忆：{name}"
        current = append_assistant_message(current, assistant_text)
        return AgentRunResult(
            state=current,
            terminal=TerminalState.AWAITING_USER,
            response={"assistant_message": assistant_text},
        )

    skill_name = _skill_name_from_message(event.message)
    if skill_name:
        register_all_bundled_skills()
        skill = get_skill(skill_name)
        if skill is not None and skill.user_invocable:
            assistant_text = await skill.execute(current, executor)
            current = append_assistant_message(current, assistant_text)
            return AgentRunResult(
                state=current,
                terminal=TerminalState.AWAITING_USER,
                response={"assistant_message": assistant_text},
            )

    # Step 1: extraction
    extract_result = await executor.execute(
        "extract_answers.deepseek",
        ExtractAnswersDeepSeekInput(messages=current.messages),
        context=ctx,
    )
    if extract_result.error is None and isinstance(extract_result.data, ExtractionResult):
        current = apply_extraction_result(current, extract_result.data)
    else:
        current = current.model_copy(deep=True)
        current.validation_errors.append("extraction: AI 提取失败")

    # Step 2: readiness
    readiness_result = await executor.execute(
        "readiness.check",
        ReadinessCheckInput(answers=current.answers, branch=current.branch),
        context=ctx,
    )
    readiness: ReadinessResult | None = None
    if readiness_result.error is None and isinstance(readiness_result.data, ReadinessResult):
        readiness = readiness_result.data
        current = current.model_copy(deep=True)
        current.readiness_result = readiness

    # Step 3: memory recall
    memory_entries = await _recall_memory_entries(executor, current, event.message)

    # Step 4: dialogue (unified input builder)
    dialogue_input = _build_dialogue_input(current, readiness, memory_entries)
    dialogue_result = await executor.execute(
        "dialogue.deepseek",
        dialogue_input,
        context=ctx,
    )
    if dialogue_result.error is not None:
        current = append_assistant_message(current, _safe_error(""))
        return AgentRunResult(
            state=current,
            terminal=TerminalState.FAILED,
            response={"assistant_message": _safe_error("")},
        )

    assistant_text = dialogue_result.data.assistant_message
    current = append_assistant_message(current, assistant_text)

    resp: dict = {"assistant_message": assistant_text}
    if readiness:
        resp["readiness"] = readiness.model_dump()

    return AgentRunResult(
        state=current,
        terminal=TerminalState.AWAITING_USER,
        response=resp,
    )


async def _handle_finish_requested(
    state: AgentState,
    registry: ToolRegistry,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_model: str | None = None,
    db_session=None,
) -> AgentRunResult:
    executor = ToolExecutor(registry)
    ctx = ToolContext(
        db_session=db_session,
        provider_base_url=provider_base_url,
        provider_api_key=provider_api_key,
        provider_model=provider_model,
    )

    # 1. readiness
    result = await executor.execute(
        "readiness.check",
        ReadinessCheckInput(answers=state.answers, branch=state.branch),
        context=ctx,
    )
    if result.error is not None:
        return AgentRunResult(state=state, terminal=TerminalState.FAILED)
    readiness = result.data
    if not isinstance(readiness, ReadinessResult):
        return AgentRunResult(state=state, terminal=TerminalState.FAILED)

    current = state.model_copy(deep=True)
    current.readiness_result = readiness

    if readiness.unsupported_branch:
        return AgentRunResult(state=current, terminal=TerminalState.UNSUPPORTED_BRANCH)
    if not readiness.score_ready:
        # Recovery: full-history extraction + re-check readiness
        recovery_result = await executor.execute(
            "extract_answers.deepseek",
            ExtractAnswersDeepSeekInput(messages=state.messages, history_window=None),
            context=ctx,
        )
        if recovery_result.error is None and isinstance(recovery_result.data, ExtractionResult):
            current = apply_extraction_result(current, recovery_result.data)
            # re-check
            retry_readiness = await executor.execute(
                "readiness.check",
                ReadinessCheckInput(answers=current.answers, branch=current.branch),
                context=ctx,
            )
            if retry_readiness.error is None and isinstance(retry_readiness.data, ReadinessResult):
                current = current.model_copy(deep=True)
                current.readiness_result = retry_readiness.data
                readiness = retry_readiness.data

        if readiness.unsupported_branch:
            return AgentRunResult(state=current, terminal=TerminalState.UNSUPPORTED_BRANCH)
        if not readiness.score_ready:
            return AgentRunResult(
                state=current,
                terminal=TerminalState.MISSING_INFO,
                response={
                    "missing_items": [m.model_dump() for m in readiness.missing_items],
                    "next_questions": readiness.next_questions,
                },
            )

    # 2. score
    from app.agent.tools.local.scoring import ScoreCalculateInput
    score_result = await executor.execute(
        "score.calculate",
        ScoreCalculateInput(answers=current.answers, branch=current.branch, q1_slots=current.slots),
        context=ctx,
    )
    if score_result.error is not None:
        return AgentRunResult(state=current, terminal=TerminalState.FAILED)
    current = current.model_copy(deep=True)
    current.scoring_result = score_result.data.scoring_result

    # 3. RAG
    rag_query_parts = []
    if current.slots.industry and current.slots.industry.value:
        rag_query_parts.append(str(current.slots.industry.value))
    if current.slots.main_product and current.slots.main_product.value:
        rag_query_parts.append(str(current.slots.main_product.value))
    if current.slots.target_market and current.slots.target_market.value:
        rag_query_parts.append(str(current.slots.target_market.value))
    rag_query = " ".join(rag_query_parts) if rag_query_parts else "B2B工厂出海"

    from app.agent.tools.external.rag import RagSearchInput
    rag_result = await executor.execute("rag.search", RagSearchInput(query=rag_query), context=ctx)
    if rag_result.error is not None:
        logger.warning("rag.search failed: %s %s", rag_result.error.code, rag_result.error.message)
    rag_context = rag_result.data.matches if rag_result.error is None else []

    # 召回兜底落地：两轮后仍低 → 拒答。此时**不把低分片段交给报告生成**，
    # 并在 state 上留痕，供报告环节与审计判断"知识库未命中"。
    if rag_result.error is None and rag_result.data is not None and rag_result.data.refused:
        rag_context = []
        current = current.model_copy(deep=True)
        current.validation_errors.append(
            f"knowledge_insufficient: {rag_result.data.reason}"
        )
        logger.warning(
            "rag.search refused (best=%.3f, rounds=%d, rewrites=%s)",
            rag_result.data.best_score, rag_result.data.rounds, rag_result.data.rewrites,
        )

    # 保存安全 RAG 引用到 state（供报告详情展示）
    if rag_result.error is None and rag_result.data is not None and not rag_result.data.refused:
        from app.schemas.rag import RagMatchSafe
        safe_matches = [
            RagMatchSafe.from_match(m).model_dump() for m in (rag_result.data.matches or [])
        ]
        current = current.model_copy(deep=True)
        current.rag_matches = safe_matches

    # 4. report generation loop (max 3 attempts)
    from app.agent.tools.external.report_generation import ReportGenerateInput
    from app.agent.tools.external.report_audit import ReportAuditInput
    from app.agent.tools.local.report_tools import ReportSplitInput as RSplitInput
    from app.agent.tools.local.report_tools import ReportGuardInput as RGuardInput
    from app.reports.template_report import build_template_raw_report
    from app.schemas.audit import ReportAuditResult

    MAX_REPORT_ATTEMPTS = 3
    audit_feedback: list[str] = []
    next_escalated = False
    length_escalated_used = False
    raw_report = None
    user_report = None
    lead_report = None
    audit = None
    audit_passed = False
    current = current.model_copy(deep=True)
    current.report_retry_count = 0

    for attempt in range(1, MAX_REPORT_ATTEMPTS + 1):
        escalated = next_escalated
        next_escalated = False

        gen_result = await executor.execute(
            "report.generate.deepseek",
            ReportGenerateInput(
                state=current, rag_context=rag_context,
                audit_feedback=audit_feedback, escalated=escalated,
            ),
            context=ctx,
        )
        if gen_result.error is not None:
            logger.warning("report.generate.deepseek attempt=%d failed: %s %s",
                           attempt, gen_result.error.code, gen_result.error.message[:200])
            if attempt < MAX_REPORT_ATTEMPTS and gen_result.error.code in (
                ToolErrorCode.TRANSIENT, ToolErrorCode.LENGTH_EXCEEDED,
                ToolErrorCode.STRUCTURED_OUTPUT_ERROR,
            ):
                current.report_retry_count = attempt
                if gen_result.error.code == ToolErrorCode.LENGTH_EXCEEDED:
                    audit_feedback = [gen_result.error.message]
                    if not length_escalated_used:
                        next_escalated = True
                        length_escalated_used = True
                        continue
                    # 已用过一次 length 升级 → 不再 retry，走模板兜底
                    break
                elif gen_result.error.code == ToolErrorCode.STRUCTURED_OUTPUT_ERROR:
                    audit_feedback = [gen_result.error.message]
                # TRANSIENT: 不改 audit_feedback, next_escalated stays False
                continue
            break

        raw_report = gen_result.data.raw_report

        # split via ToolExecutor
        split_result = await executor.execute(
            "report.split",
            RSplitInput(raw_report=raw_report, scoring_result=current.scoring_result, slots=current.slots),
            context=ctx,
        )
        if split_result.error is not None:
            logger.warning("report.split attempt=%d failed: %s %s",
                           attempt, split_result.error.code, split_result.error.message)
            current.report_retry_count = attempt
            continue

        user_report = split_result.data.user_report
        lead_report = split_result.data.lead_report

        # guard via ToolExecutor
        guard_result = await executor.execute(
            "report.guard",
            RGuardInput(user_report=user_report),
            context=ctx,
        )
        if guard_result.error is not None:
            logger.warning("report.guard attempt=%d failed: %s %s",
                           attempt, guard_result.error.code, guard_result.error.message)
            audit_feedback = [guard_result.error.message]
            current.report_retry_count = attempt
            continue

        # audit
        audit_result = await executor.execute(
            "report.audit.deepseek",
            ReportAuditInput(
                user_report=user_report,
                lead_report=lead_report,
                raw_report=raw_report,
            ),
            context=ctx,
        )
        if audit_result.error is not None:
            logger.warning("report.audit.deepseek attempt=%d failed: %s %s",
                           attempt, audit_result.error.code, audit_result.error.message)
            current.report_retry_count = attempt
            continue

        audit = audit_result.data.audit_result
        current.report_retry_count = attempt

        if audit.passed and not audit.rewrite_required and audit.severity in ("pass", "warning"):
            audit_passed = True
            break

        logger.warning("report.audit rejected attempt=%d severity=%s issues=%s",
                       attempt, audit.severity, audit.issues)
        if attempt < MAX_REPORT_ATTEMPTS:
            audit_feedback = audit.issues

    # 5. 判断结果
    current = current.model_copy(deep=True)
    if audit_passed and raw_report is not None:
        current.raw_report = raw_report
        current.user_report = user_report
        current.lead_report = lead_report
        current.audit_result = audit
        current.status = "completed"
        current.used_template_report = False
        return AgentRunResult(
            state=current,
            terminal=TerminalState.COMPLETED,
        )

    # 模板兜底
    logger.warning("entering template fallback, report_retry_count=%d", current.report_retry_count)
    try:
        if current.scoring_result is None:
            return AgentRunResult(state=current, terminal=TerminalState.FAILED)
        raw = build_template_raw_report(current)

        split_r = await executor.execute(
            "report.split",
            RSplitInput(raw_report=raw, scoring_result=current.scoring_result, slots=current.slots),
            context=ctx,
        )
        if split_r.error is not None:
            return AgentRunResult(state=current, terminal=TerminalState.FAILED)

        guard_r = await executor.execute("report.guard", RGuardInput(user_report=split_r.data.user_report), context=ctx)
        if guard_r.error is not None:
            return AgentRunResult(state=current, terminal=TerminalState.FAILED)

        current.raw_report = raw
        current.user_report = split_r.data.user_report
        current.lead_report = split_r.data.lead_report
        current.used_template_report = True
        current.status = "completed"
        current.audit_result = ReportAuditResult(
            passed=True, issues=["使用模板兜底"], rewrite_required=False, severity="warning",
        )
        return AgentRunResult(
            state=current,
            terminal=TerminalState.COMPLETED_WITH_TEMPLATE,
        )
    except Exception:
        return AgentRunResult(state=current, terminal=TerminalState.FAILED)


def _trace_dict(step: str, status: str, elapsed_ms: int | None = None, summary: str | None = None) -> dict:
    return AgentTraceEvent(
        step=step, status=status, elapsed_ms=elapsed_ms, summary=summary,
    ).model_dump()


async def run_agent_event_stream(
    state: AgentState,
    event: AgentEvent,
    registry: ToolRegistry | None = None,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_model: str | None = None,
    db_session=None,
) -> AsyncGenerator[dict, None]:
    if event.type != "user_message":
        result = await run_agent_event(
            state,
            event,
            registry,
            provider_base_url=provider_base_url,
            provider_api_key=provider_api_key,
            provider_model=provider_model,
            db_session=db_session,
        )
        client_state = ConversationClientState.from_agent_state(result.state)
        yield {"type": "error", "message": "streaming 仅支持 user_message", "state": client_state.model_dump()}
        return

    tools = registry or await _build_tool_registry(db_session)
    executor = ToolExecutor(tools)
    ctx = ToolContext(
        db_session=db_session,
        provider_base_url=provider_base_url,
        provider_api_key=provider_api_key,
        provider_model=provider_model,
    )
    current = append_user_message(state, event.message)

    remember = _parse_remember_command(event.message)
    if remember is not None:
        from app.schemas.memory import MemorySaveInput
        name, content = remember
        save_result = await executor.execute(
            "memory.save",
            MemorySaveInput(
                name=name,
                description=f"用户手动保存的长期记忆：{name}",
                type="user",
                content=content,
            ),
            context=ctx,
        )
        assistant_text = (
            "记忆保存失败，请检查内容后重试"
            if save_result.error is not None
            else f"已保存到长期记忆：{name}"
        )
        current = append_assistant_message(current, assistant_text)
        cc = ConversationClientState.from_agent_state(current)
        yield {"type": "delta", "content": assistant_text}
        yield {"type": "done", "state": cc.model_dump()}
        return

    skill_name = _skill_name_from_message(event.message)
    if skill_name:
        register_all_bundled_skills()
        skill = get_skill(skill_name)
        if skill is not None and skill.user_invocable:
            assistant_text = await skill.execute(current, executor)
            current = append_assistant_message(current, assistant_text)
            cc = ConversationClientState.from_agent_state(current)
            yield {"type": "delta", "content": assistant_text}
            yield {"type": "done", "state": cc.model_dump()}
            return

    # ── extraction ──
    t0 = _time.monotonic()
    yield _trace_dict("extract", "started")
    extract_result = await executor.execute(
        "extract_answers.deepseek",
        ExtractAnswersDeepSeekInput(messages=current.messages),
        context=ctx,
    )
    extract_elapsed = int((_time.monotonic() - t0) * 1000)
    if extract_result.error is None and isinstance(extract_result.data, ExtractionResult):
        current = apply_extraction_result(current, extract_result.data)
        yield _trace_dict("extract", "completed", elapsed_ms=extract_elapsed, summary="已抽取企业画像")
    else:
        current = current.model_copy(deep=True)
        current.validation_errors.append("extraction: AI 提取失败")
        yield _trace_dict("extract", "failed", elapsed_ms=extract_elapsed, summary="信息抽取失败")

    # ── readiness ──
    t1 = _time.monotonic()
    yield _trace_dict("readiness", "started")
    readiness_result = await executor.execute(
        "readiness.check",
        ReadinessCheckInput(answers=current.answers, branch=current.branch),
        context=ctx,
    )
    readiness_elapsed = int((_time.monotonic() - t1) * 1000)
    readiness: ReadinessResult | None = None
    if readiness_result.error is None and isinstance(readiness_result.data, ReadinessResult):
        readiness = readiness_result.data
        current = current.model_copy(deep=True)
        current.readiness_result = readiness
        missing_count = len(readiness.missing_items)
        yield _trace_dict("readiness", "completed", elapsed_ms=readiness_elapsed,
                          summary=f"score_ready={readiness.score_ready}, report_ready={readiness.report_ready}, 缺失 {missing_count} 项")
    else:
        yield _trace_dict("readiness", "failed", elapsed_ms=readiness_elapsed, summary="完整度判断失败")

    # ── memory recall ──
    t2 = _time.monotonic()
    yield _trace_dict("memory_recall", "started")
    memory_entries, mem_failed = await _recall_memory_entries_with_trace_status(
        executor, current, event.message,
    )
    mem_elapsed = int((_time.monotonic() - t2) * 1000)
    if mem_failed:
        yield _trace_dict("memory_recall", "failed", elapsed_ms=mem_elapsed, summary="记忆召回失败")
        memory_entries = []
    else:
        yield _trace_dict("memory_recall", "completed", elapsed_ms=mem_elapsed,
                          summary=f"召回 {len(memory_entries)} 条记忆")

    # ── dialogue ──
    dialogue_stream_input = _build_dialogue_input(current, readiness, memory_entries)
    system_prompt = _build_dialogue_prompt(dialogue_stream_input)
    settings = get_settings()
    llm_messages = [LLMMessage(role="system", content=system_prompt)]
    for msg in current.messages[-settings.DIALOGUE_HISTORY_WINDOW:]:
        llm_messages.append(LLMMessage(role=msg.role, content=msg.content))

    t3 = _time.monotonic()
    yield _trace_dict("dialogue", "started")
    assistant_text = ""
    try:
        client_kwargs: dict = {}
        if provider_base_url: client_kwargs["base_url"] = provider_base_url
        if provider_api_key: client_kwargs["api_key"] = provider_api_key
        if provider_model: client_kwargs["model"] = provider_model
        client = DeepSeekClient(**client_kwargs)
        async for chunk in client.stream_chat(
            llm_messages,
            max_tokens=settings.DIALOGUE_MAX_TOKENS,
            temperature=settings.DIALOGUE_TEMPERATURE,
        ):
            assistant_text += chunk
            yield {"type": "delta", "content": chunk}
    except Exception as e:
        logger.warning("stream dialogue failed: %s", e)
        yield _trace_dict("dialogue", "failed", elapsed_ms=int((_time.monotonic() - t3) * 1000), summary="追问生成失败")
        cc = ConversationClientState.from_agent_state(current)
        yield {"type": "error", "message": _safe_error(""), "state": cc.model_dump()}
        return

    dialogue_elapsed = int((_time.monotonic() - t3) * 1000)

    if not assistant_text.strip():
        logger.warning("stream dialogue returned empty content")
        yield _trace_dict("dialogue", "failed", elapsed_ms=dialogue_elapsed, summary="追问生成为空")
        cc = ConversationClientState.from_agent_state(current)
        yield {"type": "error", "message": _safe_error(""), "state": cc.model_dump()}
        return

    yield _trace_dict("dialogue", "completed", elapsed_ms=dialogue_elapsed, summary=f"已生成回复 ({len(assistant_text)} 字)")

    current = append_assistant_message(current, assistant_text)
    cc = ConversationClientState.from_agent_state(current)
    yield {"type": "done", "state": cc.model_dump()}
