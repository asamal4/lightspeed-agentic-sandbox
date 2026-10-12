from __future__ import annotations

import logging

from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.types import ToolCallEvent, ToolResultEvent


def test_tool_call_logs_do_not_include_arguments(caplog) -> None:
    payload = "DO_NOT_LOG_TOOL_ARGUMENT"
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")

    EventLogger("execution").log(ToolCallEvent(name="Bash", input=payload))

    assert "tool_use: Bash" in caplog.text
    assert payload not in caplog.text


def test_tool_result_logs_do_not_include_content(caplog) -> None:
    payload = "DO_NOT_LOG_TOOL_RESULT"
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")

    EventLogger("execution").log(ToolResultEvent(output=payload))

    assert "[provider:execution] tool_result" in caplog.messages
    assert payload not in caplog.text


def test_long_tool_result_logs_no_content(caplog) -> None:
    payload = "DO_NOT_LOG_LONG_TOOL_RESULT" + "x" * 2_000
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")

    EventLogger("execution").log(ToolResultEvent(output=payload))

    assert "[provider:execution] tool_result" in caplog.text
    assert "DO_NOT_LOG_LONG_TOOL_RESULT" not in caplog.text
    assert payload[-100:] not in caplog.text
