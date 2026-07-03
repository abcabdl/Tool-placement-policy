"""
Utility functions for Dolphin MCP.
"""

import os
import sys
import json
import yaml # Added for YAML support
import logging
import dotenv
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("dolphin_mcp")
logger.setLevel(logging.CRITICAL)

# Load environment variables
dotenv.load_dotenv(override=True)

import openai
def chat_azure(prompt):
    # base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    model_name = "gpt-4o-2024-11-20"
    max_tokens = 1024  # range: [0, 4096]
    client = openai.AzureOpenAI(
        azure_endpoint=api_base,
        api_version=api_version,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.model_dump()["choices"][0]["message"]["content"]


async def load_config_from_file(config_path: str) -> dict:
    """
    Load configuration from a JSON or YAML file.
    The file type is determined by the extension (.json or .yml/.yaml).
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration
        
    Raises:
        SystemExit: If the file is not found, has an unsupported extension, or contains invalid data.
    """
    try:
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                return json.load(f)
            elif config_path.endswith(".yml") or config_path.endswith(".yaml"):
                return yaml.safe_load(f)
            else:
                print(f"Error: Unsupported configuration file extension for {config_path}. Please use .json, .yml, or .yaml.")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {config_path}.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in {config_path}: {e}")
        sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Tuple containing (chosen_model, user_query, quiet_mode, chat_mode, interactive_mode, config_path, mcp_config_path, log_messages_path, max_pages)
    """
    args = sys.argv[1:]
    chosen_model = None
    quiet_mode = False
    chat_mode = False
    interactive_mode = False  # Added interactive_mode
    config_path = "config.yml"  # default
    mcp_config_path = "mcp.json" # default
    log_messages_path = None
    max_pages = 30
    user_query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--model":
            if i + 1 < len(args):
                chosen_model = args[i+1]
                i += 2
            else:
                print("Error: --model requires an argument")
                sys.exit(1)
        elif args[i] == "--quiet":
            quiet_mode = True
            i += 1
        elif args[i] == "--chat":
            chat_mode = True
            i += 1
        elif args[i] == "--interactive" or args[i] == "-i":  # Added interactive mode check
            interactive_mode = True
            i += 1
        elif args[i] == "--config":
            if i + 1 < len(args):
                config_path = args[i+1]
                i += 2
            else:
                print("Error: --config requires an argument")
                sys.exit(1)
        elif args[i] == "--log-messages":
            if i + 1 < len(args):
                log_messages_path = args[i+1]
                i += 2
            else:
                print("Error: --log-messages requires an argument")
                sys.exit(1)
        elif args[i] == "--mcp-config":
            if i + 1 < len(args):
                mcp_config_path = args[i+1]
                i += 2
            else:
                print("Error: --mcp-config requires an argument")
                sys.exit(1)
        elif args[i] == "--max-pages":
            if i + 1 < len(args):
                try:
                    max_pages = int(args[i+1])
                except ValueError:
                    print("Error: --max-pages requires an integer argument")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --max-pages requires an argument")
                sys.exit(1)
        elif args[i] == "--help" or args[i] == "-h":
            # Skip help flags as they're handled in the main function
            i += 1
        else:
            user_query_parts.append(args[i])
            i += 1

    user_query = " ".join(user_query_parts)
    return chosen_model, user_query, quiet_mode, chat_mode, interactive_mode, config_path, mcp_config_path, log_messages_path, max_pages
