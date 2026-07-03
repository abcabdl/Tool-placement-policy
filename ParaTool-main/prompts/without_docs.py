from prompts.with_docs import STB_SYSTEM_PROMPT as _STB_WITH_DOCS


BFCL_SYSTEM_PROMPT_TEMPLATE = """
You are an expert in composing function calls.

You are given a question and a list of function names that you can call. For each function, only its name and parameter names are provided; no natural language descriptions or detailed schemas.

Available function signatures (you MUST choose only from these; do not invent new names):
{available_tool_names}

Rules:
- Return only the Python function call(s); do not include any other text.
- Format strictly as: [func1(arg1=val1, arg2=val2), func2(arg=val)].
- Use named arguments with concrete literal values only (no placeholders).
- Call multiple functions if needed, ordered logically.
"""


STB_SYSTEM_PROMPT = _STB_WITH_DOCS.rstrip() + "\n\nAvailable API names:\n"
