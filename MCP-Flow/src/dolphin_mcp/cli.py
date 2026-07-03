"""
Command-line interface for Dolphin MCP.
"""

import asyncio
import sys
import logging
from .utils import parse_arguments, load_config_from_file # Added load_config_from_file
from .client import run_interaction, MCPAgent # Added MCPAgent

async def main(): # Changed to async def
    """
    Main entry point for the CLI.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level to DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Log to stderr
    )
    logger = logging.getLogger("dolphin_mcp") # Get logger instance after basicConfig
    logger.debug("Logging configured at DEBUG level.")

    # Check for help flag first
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: dolphin-mcp-cli [--model <name>] [--quiet] [--interactive | -i] [--config <file>] [--mcp-config <file>] [--log-messages <file>] [--debug] ['your question']")
        print("\nOptions:")
        print("  --model <name>         Specify the model to use")
        print("  --quiet                Suppress intermediate output (except errors)")
        print("  --interactive, -i      Enable interactive chat mode")
        print("  --config <file>        Specify a custom config file for providers (default: config.yml)")
        print("  --mcp-config <file>    Specify a custom config file for MCP servers (default: examples/sqlite-mcp.json)")
        print("  --log-messages <file>  Log all LLM interactions to a JSONL file")
        # Consider adding a --debug flag later if needed, but basicConfig sets it for now
        print("  --help, -h             Show this help message")
        sys.exit(0)

    chosen_model_name, user_query, quiet_mode, chat_mode, interactive_mode, config_path, mcp_config_path, log_messages_path = parse_arguments()

    if interactive_mode:
        logger.debug("Interactive mode enabled.")
        provider_config = await load_config_from_file(config_path)
        agent = await MCPAgent.create(
            model_name=chosen_model_name,
            provider_config=provider_config,
            mcp_server_config_path=mcp_config_path,
            quiet_mode=quiet_mode, # Pass quiet_mode
            log_messages_path=log_messages_path,
            stream=True # Interactive mode implies streaming
        )
        loop = asyncio.get_event_loop()
        try:
            while True:
                current_query = ""
                if user_query: # Use initial query first if provided
                    current_query = user_query
                    print(f"> {current_query}") # Simulate user typing the initial query
                    user_query = None # Clear after use
                else:
                    try:
                        user_input = await loop.run_in_executor(None, input, "> ")
                    except EOFError: # Handle Ctrl+D
                        print("\nExiting interactive mode.")
                        break 
                    if user_input.lower() in ["exit", "quit"]:
                        print("Exiting interactive mode.")
                        break
                    if not user_input:
                        continue
                    current_query = user_input
                
                if not current_query.strip(): # If after all that, query is empty, continue
                    continue

                if not quiet_mode:
                    print("AI: ", end="", flush=True)

                response_generator = await agent.prompt(current_query)
                full_response = ""
                async for chunk in response_generator:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print() # Add a newline after the full response
                
                # In a real chat, we might add full_response to a history
                # For now, each input is a new prompt in the same session

        finally:
            await agent.cleanup()
            logger.debug("Agent cleaned up.")

    else: # Non-interactive mode
        if not user_query and not chat_mode: # Original condition for non-interactive query requirement
            # If not in interactive mode, and no query is provided, and not in single-shot chat mode, show usage and exit.
            print("Usage: dolphin-mcp-cli [--model <name>] [--quiet] [--config <file>] [--mcp-config <file>] [--log-messages <file>] 'your question'", file=sys.stderr)
            sys.exit(1)

        # We do not pass a config object; we pass provider_config_path and mcp_server_config_path
        final_text = await run_interaction( # Changed to await
            user_query=user_query,
            model_name=chosen_model_name,
            provider_config_path=config_path,
            mcp_server_config_path=mcp_config_path,
            quiet_mode=quiet_mode,
            # chat_mode from args determines if single-shot interaction should stream
            stream=chat_mode, 
            log_messages_path=log_messages_path
        )

        if not quiet_mode or final_text:
            print("\n" + final_text.strip() + "\n")

if __name__ == "__main__":
    asyncio.run(main()) # Changed to asyncio.run(main())
