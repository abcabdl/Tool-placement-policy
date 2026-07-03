"""
LMStudio provider implementation for Dolphin MCP.

This module enables integration with LM Studio models for inference and tool use,
providing standardized interfaces to LMStudio's Python SDK.
"""

import json
# import logging # Removed logging import
import traceback
import inspect
import sys # Added for print to stderr
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable

# LMStudio imports - core API for LM models
import lmstudio as lms
from lmstudio import Chat

# Constants
DEFAULT_MODEL_SELECTION = None  # Use default model in LMStudio

# Configure logging
# logger = logging.getLogger("dolphin_mcp") # Removed logger setup

# Type definitions
MessageType = Dict[str, Any]
FunctionDefType = Dict[str, Any]
ModelConfigType = Dict[str, Any]


async def generate_with_lmstudio(
    conversation: List[MessageType],
    model_cfg: ModelConfigType,
    all_functions: List[FunctionDefType],
    stream: bool = False
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Generate text using LMStudio's SDK.

    Args:
        conversation: The conversation history in message format
        model_cfg: Configuration for the model including name and parameters
        all_functions: Available functions for the model to call as JSON schema
        stream: Whether to stream the response (not currently supported by LMStudio)

    Returns:
        Dict containing assistant_text and tool_calls, or an AsyncGenerator for streaming
    """
    print(f"[DEBUG] --- Entering generate_with_lmstudio ---", file=sys.stderr)
    print(f"[DEBUG] Received conversation: {json.dumps(conversation, indent=2)}", file=sys.stderr)
    print(f"[DEBUG] Received model_cfg: {model_cfg}", file=sys.stderr)
    print(f"[DEBUG] Received all_functions: {json.dumps(all_functions, indent=2)}", file=sys.stderr)
    print(f"[DEBUG] Stream mode: {stream}", file=sys.stderr)

    # Streaming is not supported by LMStudio SDK yet
    if stream:
        print("[WARN] Streaming requested but not supported by LMStudio provider.", file=sys.stderr)
        print(f"[DEBUG] --- Exiting generate_with_lmstudio (streaming not supported) ---", file=sys.stderr)
        return {"assistant_text": "Streaming not supported by LMStudio provider", "tool_calls": []}

    try:
        # Get model configuration
        model_name = model_cfg.get("model", DEFAULT_MODEL_SELECTION)
        print(f"[DEBUG] Resolved LMStudio model name: {model_name}", file=sys.stderr)

        # Initialize the model
        print(f"[DEBUG] Initializing LMStudio model: lms.llm('{model_name}')", file=sys.stderr)
        model = lms.llm(model_name)
        print(f"[DEBUG] LMStudio model initialized: {model}", file=sys.stderr)

        # Get the last user message for the prompt
        print("[DEBUG] Extracting last user message...", file=sys.stderr)
        user_message = extract_last_user_message(conversation)
        if not user_message:
             print("[WARN] No user message found in conversation history.", file=sys.stderr)
             print(f"[DEBUG] --- Exiting generate_with_lmstudio (no user message) ---", file=sys.stderr)
             return {"assistant_text": "No user message provided.", "tool_calls": []}
        print(f"[DEBUG] Extracted user message: '{user_message}'", file=sys.stderr)

        # Extract system message for context
        print("[DEBUG] Extracting system message...", file=sys.stderr)
        system_message = extract_system_message(conversation)
        print(f"[DEBUG] Extracted system message: '{system_message}'", file=sys.stderr)

        # If there are functions, use the tool interface (model.act)
        if all_functions and len(all_functions) > 0:
            print(f"[INFO] Tool use detected. Preparing {len(all_functions)} functions for model.act.", file=sys.stderr)

            # Store tool calls here
            tool_calls = []
            print("[DEBUG] Initialized empty tool_calls list.", file=sys.stderr)

            # Create callable Python functions from MCP function definitions
            python_functions = []
            print("[DEBUG] Starting creation of Python function wrappers...", file=sys.stderr)
            for i, func_def in enumerate(all_functions):
                print(f"[DEBUG] Processing function definition {i+1}/{len(all_functions)}: {func_def.get('name')}", file=sys.stderr)
                # Use the standard docstring function creator
                py_func = create_python_function_standard_docstring(func_def, tool_calls)
                if py_func is not None:
                    python_functions.append(py_func)
                    print(f"[DEBUG] Successfully created and added wrapper for: {py_func.__name__}", file=sys.stderr)
                else:
                    print(f"[WARN] Failed to create wrapper for function: {func_def.get('name')}", file=sys.stderr)
            print(f"[DEBUG] Finished creating wrappers. Total created: {len(python_functions)}", file=sys.stderr)

            # Log the prepared functions
            if python_functions:
                print("[DEBUG] Functions prepared for model.act:", file=sys.stderr)
                for f in python_functions:
                    print(f"[DEBUG]   - Name: {f.__name__}, Docstring: '{f.__doc__}'", file=sys.stderr)
            else:
                print("[WARN] No valid Python function wrappers were created for tool use.", file=sys.stderr)
                # Decide if we should proceed without tools or return an error.
                # For now, let's proceed as if no tools were requested.
                # This might need adjustment based on desired behavior.
                print("[WARN] Proceeding with regular chat inference as no tools could be prepared.", file=sys.stderr)
                # Fall through to the 'else' block below

            # Only proceed with model.act if we have functions
            if python_functions:
                # Create a clean chat object *without* system prompt for model.act's on_message
                print("[DEBUG] Creating Chat() object for model.act on_message callback.", file=sys.stderr)
                response_chat = Chat()

                # Execute model with tools using model.act()
                print(f"[INFO] Calling model.act(prompt='{user_message[:100]}...', tools=[...], on_message=...)", file=sys.stderr)
                try:
                    model.act(
                        user_message,           # Pass only the user prompt
                        python_functions,       # Pass the list of Python functions
                        on_message=response_chat.append # Collect all messages
                    )
                    print("[INFO] model.act() completed successfully.", file=sys.stderr)
                except Exception as act_error:
                     print(f"[ERROR] Exception during model.act(): {act_error}", file=sys.stderr)
                     print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr)
                     response_text = str(response_chat) # Get any text collected before error
                     print(f"[DEBUG] Text collected by response_chat before error: {response_text}", file=sys.stderr)
                     print(f"[DEBUG] Tool calls collected before error: {tool_calls}", file=sys.stderr)
                     print(f"[DEBUG] --- Exiting generate_with_lmstudio (model.act error) ---", file=sys.stderr)
                     return {
                        "assistant_text": f"Error during tool execution: {act_error}\nCollected text: {response_text}",
                        "tool_calls": tool_calls
                     }

                # Get the full text response collected by the chat
                response_text = str(response_chat)
                print(f"[DEBUG] Full response collected by response_chat after model.act(): {response_text}", file=sys.stderr)
                print(f"[DEBUG] Final tool_calls collected: {tool_calls}", file=sys.stderr)

                # Return the collected text and any captured tool calls
                result = {
                    "assistant_text": response_text, # Return the full chat history string for now
                    "tool_calls": tool_calls # Populated by the wrapper functions
                }
                print(f"[DEBUG] --- Exiting generate_with_lmstudio (tool mode success) ---", file=sys.stderr)
                return result

        # Regular chat without tools (or if tool prep failed)
        print("[INFO] Using regular chat inference (model.respond).", file=sys.stderr)
        print(f"[DEBUG] Creating Chat object. System message provided: {bool(system_message)}", file=sys.stderr)
        chat = Chat(system_message) if system_message else Chat()
        print(f"[DEBUG] Adding user message to chat: '{user_message}'", file=sys.stderr)
        chat.add_user_message(user_message)
        print(f"[DEBUG] Chat history before respond: {str(chat)}", file=sys.stderr)

        print("[INFO] Calling model.respond(chat)...", file=sys.stderr)
        response = model.respond(chat)
        print("[INFO] model.respond() completed.", file=sys.stderr)
        print(f"[DEBUG] Raw response from model.respond(): {response}", file=sys.stderr)

        result = {
            "assistant_text": response,
            "tool_calls": []
        }
        print(f"[DEBUG] --- Exiting generate_with_lmstudio (respond mode success) ---", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[ERROR] Unhandled exception in generate_with_lmstudio: {str(e)}", file=sys.stderr)
        print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr)
        print(f"[DEBUG] --- Exiting generate_with_lmstudio (unhandled exception) ---", file=sys.stderr)
        return {"assistant_text": f"LMStudio provider error: {str(e)}", "tool_calls": []}


def extract_last_user_message(conversation: List[MessageType]) -> str:
    """Extract the last user message from the conversation."""
    print("[DEBUG] --- Entering extract_last_user_message ---", file=sys.stderr)
    for i, message in enumerate(reversed(conversation)):
        print(f"[DEBUG] Checking message {-i-1}: role={message.get('role')}", file=sys.stderr)
        if message.get("role") == "user":
            content = message.get("content", "")
            print(f"[DEBUG] Found user message. Content type: {type(content)}", file=sys.stderr)
            if isinstance(content, list):
                content_text = ""
                for j, part in enumerate(content):
                    print(f"[DEBUG] Processing content part {j}: type={part.get('type')}", file=sys.stderr)
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_part = part.get("text", "")
                        content_text += text_part
                        print(f"[DEBUG] Added text part: '{text_part[:50]}...'", file=sys.stderr)
                print(f"[DEBUG] Extracted text from list content: '{content_text[:100]}...'", file=sys.stderr)
                print("[DEBUG] --- Exiting extract_last_user_message (found list) ---", file=sys.stderr)
                return content_text
            elif isinstance(content, str):
                print(f"[DEBUG] Extracted text from string content: '{content[:100]}...'", file=sys.stderr)
                print("[DEBUG] --- Exiting extract_last_user_message (found str) ---", file=sys.stderr)
                return content
    print("[WARN] No user message found in conversation.", file=sys.stderr)
    print("[DEBUG] --- Exiting extract_last_user_message (not found) ---", file=sys.stderr)
    return "" # Return empty string if no user message found

def extract_system_message(conversation: List[MessageType]) -> str:
    """Extract the system message from the conversation."""
    print("[DEBUG] --- Entering extract_system_message ---", file=sys.stderr)
    system_message = ""
    found = False
    for i, message in enumerate(conversation):
        print(f"[DEBUG] Checking message {i}: role={message.get('role')}", file=sys.stderr)
        if message.get("role") == "system":
            found = True
            content = message.get("content", "")
            print(f"[DEBUG] Found system message. Content type: {type(content)}", file=sys.stderr)
            if isinstance(content, list):
                for j, part in enumerate(content):
                    print(f"[DEBUG] Processing content part {j}: type={part.get('type')}", file=sys.stderr)
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_part = part.get("text", "")
                        system_message += text_part + "\n"
                        print(f"[DEBUG] Added text part: '{text_part[:50]}...'", file=sys.stderr)
            elif isinstance(content, str):
                system_message += content + "\n"
                print(f"[DEBUG] Added string content: '{content[:100]}...'", file=sys.stderr)
    if not found:
        print("[DEBUG] No system message found in conversation.", file=sys.stderr)
    final_message = system_message.strip()
    print(f"[DEBUG] Final extracted system message: '{final_message[:100]}...'", file=sys.stderr)
    print("[DEBUG] --- Exiting extract_system_message ---", file=sys.stderr)
    return final_message

def map_json_type_to_python_str(json_type: str) -> str:
    """Maps JSON schema types to Python type hint strings for docstrings."""
    # No logging needed here, it's a simple mapping function.
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float", # Using float for number
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    py_type = type_map.get(json_type, "Any")
    # print(f"[DEBUG] Mapped JSON type '{json_type}' to Python type string '{py_type}'", file=sys.stderr) # Optional: uncomment if needed
    return py_type

def create_python_function_standard_docstring(func_def: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Optional[Callable]:
    """
    Create a Python function wrapper using standard parameters and docstring format.

    Args:
        func_def: The MCP function definition (includes 'name', 'description', 'parameters')
        tool_calls: A list to store captured tool calls for Dolphin MCP

    Returns:
        A callable Python function object or None if creation failed
    """
    full_name = func_def.get("name", "unknown_tool") # Get name early for logging
    print(f"[DEBUG] --- Entering create_python_function_standard_docstring for: {full_name} ---", file=sys.stderr)
    try:
        print(f"[DEBUG] Processing func_def: {json.dumps(func_def, indent=2)}", file=sys.stderr)

        name_parts = full_name.split("_", 1)
        if len(name_parts) < 2:
            print(f"[ERROR] Invalid function name format for '{full_name}'. Expected 'server_tool'.", file=sys.stderr)
            print(f"[DEBUG] --- Exiting create_python_function_standard_docstring (invalid name format) ---", file=sys.stderr)
            return None

        server_name = name_parts[0]
        function_name = name_parts[1] # Simple name for LMStudio
        print(f"[DEBUG] Parsed name -> LMStudio name: '{function_name}', Server: '{server_name}'", file=sys.stderr)

        description = func_def.get("description", "")
        parameters = func_def.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        print(f"[DEBUG] Extracted description: '{description[:50]}...'", file=sys.stderr)
        print(f"[DEBUG] Extracted properties: {list(properties.keys())}", file=sys.stderr)
        print(f"[DEBUG] Required parameters: {required}", file=sys.stderr)

        # Build a standard Python docstring
        print("[DEBUG] Building docstring...", file=sys.stderr)
        docstring_parts = [f"{description}\n"] # Description first
        if properties:
            docstring_parts.append("Args:") # Standard 'Args:' section
            for param_name, param_info in properties.items():
                param_type_json = param_info.get("type", "any")
                param_type_py_str = map_json_type_to_python_str(param_type_json)
                param_desc = param_info.get("description", "")
                # Format: param_name (type): description. (Required/Optional)
                required_str = "Required." if param_name in required else "Optional."
                docstring_parts.append(f"    {param_name} ({param_type_py_str}): {param_desc} {required_str}")
        docstring = "\n".join(docstring_parts)
        print(f"[DEBUG] Built docstring for {function_name}:\n{docstring}", file=sys.stderr)

        # Generate parameter list with type hints from JSON schema
        print("[DEBUG] Building parameter string for function definition...", file=sys.stderr)
        param_list = []
        for param_name, param_info in properties.items():
            py_type = map_json_type_to_python_str(param_info.get("type", "any"))
            if param_name in required:
                param_list.append(f"{param_name}: {py_type}")
            else:
                # Determine a sensible default based on type for the signature
                default_value_str = "None" # Default to None for optional non-strings
                if param_info.get("type") == "string":
                    default_value_str = '""'
                elif param_info.get("type") == "boolean":
                     default_value_str = "False" # Or True? False seems safer.
                elif param_info.get("type") == "integer" or param_info.get("type") == "number":
                     default_value_str = "0" # Or None? 0 might be okay.
                elif param_info.get("type") == "array":
                     default_value_str = "[]"
                elif param_info.get("type") == "object":
                     default_value_str = "{}"

                param_list.append(f"{param_name}: Optional[{py_type}] = {default_value_str}") # Use Optional for clarity
        param_str = ", ".join(param_list)
        print(f"[DEBUG] Built parameter string: '{param_str}'", file=sys.stderr)

        param_names = list(properties.keys()) # Get list of expected parameter names
        print(f"[DEBUG] List of parameter names for wrapper: {param_names}", file=sys.stderr)

        # Dynamically compile the function with proper syntax
        # Ensure only necessary variables are passed to exec_scope
        exec_scope = {
            'tool_calls': tool_calls,
            'json': json,
            'function_name': function_name, # Simple name for logging inside wrapper
            'full_name': full_name,         # Full MCP name for tool_call object
            # 'logger': logger, # Removed logger from scope
            'param_names': param_names,
            'traceback': traceback,         # Pass traceback module for logging
            'Optional': Optional,           # Make Optional available for type hints
            'sys': sys                      # Make sys available for print
        }
        print(f"[DEBUG] Preparing execution scope for exec(). Keys: {list(exec_scope.keys())}", file=sys.stderr)

        # More robust code string using locals() and explicit parameter names
        # Includes more logging within the generated function
        code_string = f'''
import json # Ensure json is available inside the function too
import traceback # Ensure traceback is available
from typing import Optional # Ensure Optional is available
import sys # For print to stderr

# Define the function with the generated parameter string
def tool_function_wrapper({param_str}):
    """Dynamically created wrapper for LMStudio tool use: {function_name}."""
    print(f"[DEBUG] --- Entering LMStudio wrapper: {{function_name}} ---", file=sys.stderr)
    args_dict = {{}} # Initialize args_dict for logging/tool_call
    try:
        # Capture arguments passed to the function using locals()
        local_vars = locals()
        print(f"[DEBUG] Wrapper {{function_name}} locals(): {{local_vars}}", file=sys.stderr)

        # Construct args_dict *only* from expected parameters defined in param_names
        # Filter out None values unless explicitly needed? For now, include them.
        args_dict = {{k: local_vars[k] for k in param_names if k in local_vars}}
        print(f"[DEBUG] Constructed args_dict for {{function_name}}: {{args_dict}}", file=sys.stderr)

        # Generate a unique-ish call ID
        # Using hash of sorted JSON args is reasonably stable
        try:
            args_json_str = json.dumps(args_dict, sort_keys=True)
            call_id_suffix = abs(hash(args_json_str))
        except TypeError: # Handle non-serializable args if they somehow occur
            print(f"[WARN] Could not serialize args for call_id generation in {{function_name}}. Using fallback.", file=sys.stderr)
            call_id_suffix = "fallback"
        call_id = f"call_{{full_name}}_{{call_id_suffix}}"
        print(f"[DEBUG] Generated call_id: {{call_id}}", file=sys.stderr)

        # Create the tool_call dictionary for Dolphin MCP
        tool_call = {{
            "id": call_id,
            "function": {{
                "name": full_name, # Use the full MCP name (server_tool)
                "arguments": json.dumps(args_dict) # Arguments as JSON string
            }}
        }}
        print(f"[DEBUG] Created tool_call object for {{full_name}}: {{tool_call}}", file=sys.stderr)

        # Safely append to tool_calls list (passed via exec_scope)
        # Ensure tool_calls is treated as a list
        if isinstance(tool_calls, list):
             tool_calls.append(tool_call)
             print(f"[INFO] Appended tool call for {{full_name}} to shared list.", file=sys.stderr)
        else:
             print(f"[ERROR] CRITICAL: tool_calls object in wrapper {{function_name}} is not a list! Type: {{type(tool_calls)}}", file=sys.stderr)


        # Return a confirmation message (LMStudio expects a string response from tools)
        result_msg = f"[Tool call {{function_name}} captured by wrapper]"
        print(f"[DEBUG] --- Exiting LMStudio wrapper {{function_name}} (success) ---", file=sys.stderr)
        return result_msg

    except Exception as e:
        # More detailed error logging inside the wrapper
        print(f"[ERROR] Exception inside LMStudio tool wrapper {{function_name}} for {{full_name}}: {{e}}", file=sys.stderr)
        print(f"[DEBUG] Arguments during exception: {{args_dict}}", file=sys.stderr)
        # Use traceback passed in scope
        print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr)
        error_msg = f"[Error capturing tool call for {{function_name}}: {{e}}]"
        print(f"[DEBUG] --- Exiting LMStudio wrapper {{function_name}} (exception) ---", file=sys.stderr)
        return error_msg # Return error message string
'''
        print(f"[DEBUG] Code string to be executed for {function_name}:\n{code_string[:500]}...", file=sys.stderr) # Log start of code string

        try:
            exec(code_string, exec_scope)
            print(f"[DEBUG] Exec completed successfully for {function_name}", file=sys.stderr)
        except Exception as exec_error:
            # Catch errors during the exec call itself (e.g., syntax errors in generated code)
            print(f"[ERROR] CRITICAL: Failed to exec function definition for {function_name}: {exec_error}", file=sys.stderr)
            print(f"[DEBUG] Code string attempted:\n{code_string}", file=sys.stderr)
            print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr)
            print(f"[DEBUG] --- Exiting create_python_function_standard_docstring (exec error) ---", file=sys.stderr)
            return None # Fail creation if exec fails

        # Retrieve the dynamically created function from the execution scope
        tool_function_wrapper = exec_scope.get('tool_function_wrapper') # Use .get() for safety
        if tool_function_wrapper is None:
             print(f"[ERROR] Failed to retrieve 'tool_function_wrapper' from exec_scope after exec for {function_name}", file=sys.stderr)
             print(f"[DEBUG] --- Exiting create_python_function_standard_docstring (retrieval failed) ---", file=sys.stderr)
             return None
        print(f"[DEBUG] Retrieved function object for {function_name}: {tool_function_wrapper}", file=sys.stderr)

        # Set the essential metadata: name and the standard docstring
        tool_function_wrapper.__name__ = function_name # Use the simple name for LMStudio
        tool_function_wrapper.__doc__ = docstring
        print(f"[DEBUG] Set __name__='{function_name}' and __doc__ for wrapper.", file=sys.stderr)

        print(f"[INFO] Successfully created Python function wrapper for {full_name} (LMStudio name: {function_name})", file=sys.stderr)
        print(f"[DEBUG] --- Exiting create_python_function_standard_docstring (success) ---", file=sys.stderr)
        return tool_function_wrapper

    except Exception as e:
        print(f"[ERROR] Unhandled exception in create_python_function_standard_docstring for {full_name}: {e}", file=sys.stderr)
        print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr)
        print(f"[DEBUG] --- Exiting create_python_function_standard_docstring (unhandled exception) ---", file=sys.stderr)
        return None # Logged failure implicitly by returning None
