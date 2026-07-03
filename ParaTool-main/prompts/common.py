ASSISTANT_PROMPT = "{answer}"


MULTISTEP_TOOLCALL_HINT = """
[Multi-step tool calling mode]

- If the question contains a [TOOLCALL_HISTORY] section listing previous calls,
  assume those calls have already been executed. Do NOT repeat them; only
  output the additional call(s) needed for the current step.
- For multi-step questions, each answer should contain only the call(s) for
  the current step, not the full history.
- Output the minimum call set for this step only. In this benchmark, that is
  usually exactly one function call per turn. Do NOT pre-emptively solve later
  subtasks from the user request, even if the original request mentions them.
- Append exactly one status token at the end:
  - use <CALL_CONT> if more tool calls will be needed in later turns;
  - use <CALL_END> if all required calls have been produced and no further
    calls are needed.
"""
