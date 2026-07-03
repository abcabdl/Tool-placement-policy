import os
import sys
import json
import re

from typing import List, Dict, Any, Tuple, Optional, Callable


from root_dir_path import ROOT_DIR

STB_ROOT = os.path.join(ROOT_DIR, "StableToolBench")
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)

from toolbench.utils import standardize, change_name  # type: ignore
from dataset.profiles import StbReactProfile


def load_tool_schema_from_toolenv(
    category_name: str,
    tool_name: str,
    tool_root_dir: str,
) -> Optional[Dict[str, Any]]:
    standard_tool_name = standardize(tool_name)

    tool_file = os.path.join(tool_root_dir, category_name, f"{standard_tool_name}.json")

    if not os.path.exists(tool_file):
        return None

    try:
        with open(tool_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fetch_api_json(
    raw_api_list: List[Dict[str, Any]],
    tool_root_dir: str,
) -> List[Dict[str, Any]]:
    api_json_list = []

    for raw_api in raw_api_list:
        category_name = raw_api.get("category_name", "")
        tool_name = raw_api.get("tool_name", "")
        api_name = raw_api.get("api_name", "")

        if not all([category_name, tool_name, api_name]):
            continue

        standard_tool_name = change_name(standardize(tool_name))
        pure_api_name = change_name(standardize(api_name))

        tool_schema = load_tool_schema_from_toolenv(
            category_name, tool_name, tool_root_dir
        )

        if tool_schema is None:
            api_json_list.append(raw_api)
            continue

        api_list = tool_schema.get("api_list", [])
        matched_api = None

        for api_dict in api_list:
            api_dict_name = api_dict.get("name", "")
            if change_name(standardize(api_dict_name)) == pure_api_name:
                matched_api = api_dict
                break

        if matched_api is None:
            api_json_list.append(raw_api)
            continue

        api_json = {
            "category_name": category_name,
            "tool_name": tool_name,
            "api_name": matched_api.get("name", api_name),

            "api_description": matched_api.get("description", ""),
            "required_parameters": matched_api.get("required_parameters", []),
            "optional_parameters": matched_api.get("optional_parameters", []),
            "method": matched_api.get("method", "GET"),
            "template_response": matched_api.get("template_response", {}),
        }

        api_json_list.append(api_json)

    return api_json_list

def convert_type_to_openai(param_type: str) -> str:
    param_type_upper = str(param_type).upper()

    if param_type_upper in ("NUMBER", "INTEGER", "INT"):
        return "integer"
    elif param_type_upper in ("STRING", "STR"):
        return "string"
    elif param_type_upper in ("BOOLEAN", "BOOL"):
        return "boolean"
    else:
        return "string"

def generate_openai_function_schema(
    api_json: Dict[str, Any],
) -> Dict[str, Any]:
    category_name = api_json.get("category_name", "")
    tool_name = api_json.get("tool_name", "")
    api_name = api_json.get("api_name", "")
    api_description = api_json.get("api_description", "")
    required_params = api_json.get("required_parameters", [])
    optional_params = api_json.get("optional_parameters", [])

    standard_tool_name = change_name(standardize(tool_name))
    pure_api_name = change_name(standardize(api_name))

    function_name = f"{pure_api_name}_for_{standard_tool_name}"
    function_name = function_name[-64:]

    properties = {}
    required_names = []

    for param in required_params:
        param_name = change_name(standardize(param.get("name", "")))
        param_type = convert_type_to_openai(param.get("type", "STRING"))
        param_desc = param.get("description", "")

        properties[param_name] = {
            "type": param_type,
            "description": param_desc,
        }
        required_names.append(param_name)

    for param in optional_params:
        param_name = change_name(standardize(param.get("name", "")))
        param_type = convert_type_to_openai(param.get("type", "STRING"))
        param_desc = param.get("description", "")

        properties[param_name] = {
            "type": param_type,
            "description": param_desc,
        }

    function_schema = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": api_description or f"API: {api_name} from {tool_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_names,
            },
        },
    }

    return function_schema

def prepare_tools_from_sample(
    raw_item: Dict[str, Any],
    tool_root_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_api_list = raw_item.get("api_list", [])

    api_json_list = fetch_api_json(raw_api_list, tool_root_dir)

    functions = []
    meta = []

    for api_json in api_json_list:
        func_schema = generate_openai_function_schema(api_json)
        functions.append(func_schema)

        category_name = api_json.get("category_name", "")
        tool_name = api_json.get("tool_name", "")
        api_name = api_json.get("api_name", "")

        standard_tool_name = change_name(standardize(tool_name))
        pure_api_name = change_name(standardize(api_name))

        meta_entry = {
            "category_name": category_name,
            "standard_tool_name": standard_tool_name,
            "pure_api_name": pure_api_name,
            "original_tool_name": tool_name,
            "original_api_name": api_name,
            "function_name": func_schema["function"]["name"],
        }
        meta.append(meta_entry)

    finish_func = {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": (
                "If you believe that you have obtained a result that can answer "
                "the task, please call this function to provide the final "
                "answer. Alternatively, if you recognize that you are unable "
                "to proceed with the task in the current state, call this "
                "function to restart. Remember: you must ALWAYS call this "
                "function at the end of your attempt, and the only part that "
                "will be shown to the user is the final answer, so it should "
                "contain sufficient information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["give_answer", "give_up_and_restart"],
                    },
                    "final_answer": {
                        "type": "string",
                        "description": (
                            "The final answer you want to give the user. "
                            "You should set this when return_type == "
                            "'give_answer'."
                        ),
                    },
                },
                "required": ["return_type"],
            },
        },
    }
    functions.append(finish_func)

    return functions, meta

def find_meta_by_function_name(
    function_name: str,
    meta_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for meta_entry in meta_list:
        if meta_entry.get("function_name") == function_name:
            return meta_entry
    return None

import requests


class ToolbenchEnv:
    def __init__(
        self,
        service_url: Optional[str] = None,
        toolbench_key: Optional[str] = None,
        strip: str = "truncate",
        timeout: int = 30,
        cache_save: Optional[bool] = None,
    ):
        self.service_url = service_url or os.environ.get(
            "STB_MIRRORAPI_URL", "http://127.0.0.1:8080/virtual"
        )
        self.toolbench_key = toolbench_key or os.environ.get(
            "STB_TOOLBENCH_KEY", "EMPTY"
        )
        if cache_save is None:
            raw = os.environ.get("STB_CACHE_IS_SAVE")
            if raw is not None:
                v = str(raw).strip().lower()
                if v in {"1", "true", "yes", "y", "on"}:
                    cache_save = True
                elif v in {"0", "false", "no", "n", "off"}:
                    cache_save = False
        self.cache_save = cache_save
        self.strip = strip
        self.timeout = timeout

        self.query = ""
        self.functions = []
        self.meta_list = []

    def reset(
        self,
        raw_item: Dict[str, Any],
        functions: List[Dict[str, Any]],
        meta_list: List[Dict[str, Any]],
    ):
        self.query = raw_item.get("query", "")
        self.functions = functions
        self.meta_list = meta_list

    def step(
        self,
        action_name: str,
        action_input_dict: Dict[str, Any],
    ) -> Tuple[str, int]:
        if action_name == "Finish":
            observation_str = '{"response": "successfully giving the final answer."}'
            status_code = 3
            return observation_str, status_code

        meta_entry = find_meta_by_function_name(action_name, self.meta_list)
        if meta_entry is None:
            observation_str = '{"error": "Unknown function", "response": ""}'
            status_code = 2
            return observation_str, status_code

        category = meta_entry["category_name"]
        tool_name = meta_entry["original_tool_name"]
        api_name = meta_entry["original_api_name"]

        payload = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "tool_input": action_input_dict,
            "strip": self.strip,
            "toolbench_key": self.toolbench_key,
        }
        if self.cache_save is not None:
            payload["is_save"] = bool(self.cache_save)

        try:
            resp = requests.post(
                self.service_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            error = data.get("error", "")
            if not error:
                observation_str = json.dumps(data, ensure_ascii=False)
                status_code = 0
            else:
                observation_str = json.dumps(data, ensure_ascii=False)
                status_code = 2

            return observation_str, status_code

        except requests.exceptions.Timeout:
            observation_str = '{"error": "Request timeout", "response": ""}'
            status_code = 2
            return observation_str, status_code

        except requests.exceptions.RequestException as e:
            observation_str = json.dumps(
                {"error": f"Request failed: {str(e)}", "response": ""},
                ensure_ascii=False,
            )
            status_code = 2
            return observation_str, status_code

        except Exception as e:
            observation_str = json.dumps(
                {"error": f"Unexpected error: {str(e)}", "response": ""},
                ensure_ascii=False,
            )
            status_code = 2
            return observation_str, status_code

def build_toolbench_question_text(
    query: str,
    history_steps: List[Dict[str, Any]],
) -> str:
    query_text = str(query or "")
    if not history_steps:
        return query_text

    lines: List[str] = [query_text, "", "Here are the history actions and observations:"]
    for step_info in history_steps:
        thought = str(step_info.get("thought") or "").strip()
        action = str(step_info.get("action") or "").strip()
        inp = step_info.get("input", "")
        if not isinstance(inp, str):
            try:
                inp = json.dumps(inp, ensure_ascii=False)
            except Exception:
                inp = str(inp)
        obs = step_info.get("observation", "")

        if thought:
            lines.append(f"Thought: {thought}")
        else:
            lines.append("Thought:")

        lines.append(f"Action: {action}")
        lines.append(f"Action Input: {inp}")
        lines.append("End Action")

        if obs:
            lines.append(f"Observation: {obs}")
        lines.append("")

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)

def run_single_query(
    raw_item: Dict[str, Any],
    env: ToolbenchEnv,
    model: Any,
    tokenizer: Any,
    generation_config: Any,
    tool_root_dir: str,
    max_steps: int = 20,
    *,
    functions: Optional[List[Dict[str, Any]]] = None,
    meta_list: Optional[List[Dict[str, Any]]] = None,
    pre_generate_hook: Optional[
        Callable[[str, List[Dict[str, Any]], int, List[Dict[str, Any]]], None]
    ] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if functions is None or meta_list is None:
        functions, meta_list = prepare_tools_from_sample(raw_item, tool_root_dir)

    env.reset(raw_item, functions, meta_list)

    query = raw_item.get("query", "")
    query_id = raw_item.get("query_id", "")
    history_steps = []
    final_answer = ""
    error_message = ""
    raw_model_outputs = []
    last_error_call_key = None
    last_error_call_repeats = 0
    hit_max_steps = False

    tools_converted = []
    for func_schema in functions:
        if isinstance(func_schema, dict) and "function" in func_schema:
            fn = func_schema["function"]
            if not isinstance(fn, dict):
                continue
            meta = find_meta_by_function_name(str(fn.get("name") or ""), meta_list)
            if meta:
                try:
                    fn = dict(fn)
                    fn.update(
                        {
                            "category_name": str(meta.get("category_name") or "").strip(),
                            "tool_name": str(meta.get("original_tool_name") or "").strip(),
                            "api_name": str(meta.get("original_api_name") or "").strip(),
                            "standard_tool_name": str(meta.get("standard_tool_name") or "").strip(),
                            "pure_api_name": str(meta.get("pure_api_name") or "").strip(),
                        }
                    )
                except Exception:
                    pass
            tools_converted.append(fn)

    profile = StbReactProfile("without")

    def _generate_with_profile(question_text: str) -> str:
        import torch

        input_ids, input_len = profile.build_inference_prompt_ids(
            tokenizer,
            question_text,
            tools_converted,
        )
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids_tensor,
                attention_mask=torch.ones(input_ids_tensor.shape).to(model.device),
                generation_config=generation_config,
            )
        if hasattr(output, "sequences"):
            output_ids = output.sequences[0][input_len:]
        else:
            output_ids = output[0][input_len:]
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    for step in range(max_steps):
        question_text = build_toolbench_question_text(query, history_steps)

        try:
            if pre_generate_hook is not None:
                try:
                    pre_generate_hook(question_text, tools_converted, int(step), history_steps)
                except Exception:
                    pass

            if verbose:
                print(f"[ToolBench][Step {step}] generating...", flush=True)
            model_output = _generate_with_profile(question_text)

            raw_model_outputs.append(model_output)

            parsed = profile.parse_inference_output(model_output, tools_converted)
            thought = ""
            action_name = ""
            action_input_raw = ""
            if isinstance(parsed, dict):
                thought = str(parsed.get("thought") or "")
                action_name = str(parsed.get("action") or "")
                action_input_raw = str(parsed.get("action_input") or "")

            if not action_name or not action_input_raw:
                error_message = "Model output missing ReAct markers"
                break

            action_name = (action_name or "").strip()
            action_input_raw = (action_input_raw or "").strip()

            # Some models emit several ReAct blocks in one generation; keep only the first action input.
            cut_pos = action_input_raw.find("End Action")
            if cut_pos != -1:
                action_input_raw = action_input_raw[:cut_pos].strip()
            else:
                fallback_markers = ["\nObservation:", "\nThought:", "\nAction:"]
                cut_pos = None
                for m in fallback_markers:
                    p = action_input_raw.find(m)
                    if p != -1:
                        cut_pos = p if cut_pos is None else min(cut_pos, p)
                if cut_pos is not None:
                    action_input_raw = action_input_raw[:cut_pos].strip()

            if "(" in action_name:
                action_name = action_name.split("(", 1)[0].strip()

            if not action_name:
                error_message = "Model returned empty action"
                break

            if verbose:
                show_inp = action_input_raw
                if len(show_inp) > 200:
                    show_inp = show_inp[:200] + "..."
                print(f"[ToolBench][Step {step}] action={action_name} input={show_inp}", flush=True)

            if action_name == "Finish":
                parsed = {}
                if action_input_raw:
                    try:
                        parsed = json.loads(action_input_raw)
                    except Exception:
                        # Recover final answers that contain raw newlines or other non-JSON text.
                        m = re.search(
                            r'"final_answer"\s*:\s*"(.*)"\s*}\s*$',
                            action_input_raw,
                            flags=re.DOTALL,
                        )
                        if m:
                            final_answer = m.group(1)
                            break
                final_answer = parsed.get("final_answer", "")
                break

            if verbose:
                print(f"[ToolBench][Step {step}] calling env.step...", flush=True)
            observation, status_code = env.step(action_name, action_input_raw)

            call_key = (action_name, action_input_raw)
            if status_code == 2:
                if call_key == last_error_call_key:
                    last_error_call_repeats += 1
                else:
                    last_error_call_key = call_key
                    last_error_call_repeats = 1
            else:
                last_error_call_key = None
                last_error_call_repeats = 0

            history_steps.append(
                {
                    "thought": thought,
                    "action": action_name,
                    "input": action_input_raw,
                    "observation": observation,
                    "code": status_code,
                }
            )

            if status_code == 3:
                break
            elif status_code == 2:
                if last_error_call_repeats >= 3:
                    error_message = (
                        f"Repeated tool error for the same call: action={action_name}"
                    )
                    break

        except Exception as e:
            error_message = f"Step {step} failed: {str(e)}"
            break
    else:
        hit_max_steps = True
        if not error_message:
            error_message = "Max steps reached without Finish"

    return {
        "query_id": query_id,
        "query": query,
        "history_steps": history_steps,
        "final_answer": final_answer,
        "error": error_message,
        "hit_max_steps": bool(hit_max_steps),
        "raw_model_outputs": raw_model_outputs,
    }
