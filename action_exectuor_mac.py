import requests
import os
import json
import time
from dotenv import load_dotenv
import typing, re
import google.generativeai as genai # Use standard import alias
from google.generativeai import types

import executions

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")

# Ensure the API key is loaded
if not GEMINI_KEY:
    raise ValueError("GEMINI_KEY not found in environment variables. Check your .env file.")

# Configure the GenAI client (do this once)
genai.configure(api_key=GEMINI_KEY)

def get_llm_interpretation(user_command: str) -> dict | None: # Add None to return type hint
    """
    Interprets a user command using the Gemini LLM and returns a plan as JSON.
    Returns None if interpretation fails or is blocked.
    """
    # Construct the full prompt dynamically
    system_prompt_template = (
        "You are a computer vision-guided execution assistant in MacOS. "
        "Based on this command, make a plan by listing out actions and parameters for executions: '{command}'  " # Placeholder for command
        "\n\n For your plan, focus on these actions:"
        "\n- 'run_app': Open an app. Parameters: 'app_name'"
        "\n- 'search': Search or look up a term. Parameters: 'term'"
        "\n- 'go_to': Go to this nth term number. Parameters: 'nth_term'"
        "\n- 'press_key': press a specific key. Parameters: 'key'"
        "\n- 'type_text': Something to type. Parameters: 'text'"
        "\n- 'wait': Wait if needed. Parameters: 'seconds'"
        "\n- 'comment': Include explanations. Parameters: 'comment_text'"
        "\n\nExample plan for 'Look up python and click the first term':"
        "[\n"
        "  {{\"action\": \"run_app\", \"term\": \"Chrome\"}}\n"
        "  {{\"action\": \"search\", \"term\": \"Python\"}}\n"
        "  {{\"action\": \"wait\", \"seconds\": \"1\"}}\n"
        "  {{\"action\": \"go_to\", \"nth_term\": \"1\"}}\n"
        "]\n"
        # Updated instruction for clarity
        "Ensure the output is ONLY the JSON list itself, without any surrounding text, comments, or markdown formatting like ```json or ```."
    )
    full_prompt = system_prompt_template.format(command=user_command)

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Using standard model

        response = model.generate_content(
            contents=[full_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                 # Explicitly request JSON output if the model supports it (Gemini doesn't directly, but helps guide)
                # response_mime_type="application/json", # This might not be supported by all models/versions, but worth trying if available
            )
            # No safety settings overridden, using defaults
        )

        # --- CRITICAL: Check the response before accessing parts ---
        # print(f"DEBUG: Full API Response:\n{response}\n---") # Uncomment for deep debugging

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"Error: Prompt was blocked. Reason: {response.prompt_feedback.block_reason}")
            return None

        if not response.candidates:
             print("Error: No candidates found in the response. Possible safety block or other issue.")
             try:
                 # Sometimes the text attribute might exist even without candidates if there was an error
                 if hasattr(response, 'text') and response.text:
                     print(f"Response text (if available): {response.text}")
             except Exception:
                 pass
             return None

        if not response.candidates[0].content or not response.candidates[0].content.parts:
             print("Error: Response candidate content or parts are missing.")
             return None

        interpretation = response.candidates[0].content.parts[0].text
        # --- End of critical checks ---

        # ----- Refined Cleanup Logic -----
        # 1. Strip leading/trailing whitespace first
        interpretation = interpretation.strip()

        # 2. Check for and remove markdown fences
        if interpretation.startswith("```json") and interpretation.endswith("```"):
            # Remove ```json from start and ``` from end
            interpretation = interpretation[len("```json"): -len("```")]
            # Strip again in case of extra newlines inside the fences
            interpretation = interpretation.strip()
        elif interpretation.startswith("```") and interpretation.endswith("```"):
             # Handle case where it might just be ``` without 'json'
             interpretation = interpretation[len("```"): -len("```")]
             interpretation = interpretation.strip()
        # ----- End of Refined Cleanup Logic -----


        if interpretation:
            try:
                # Attempt to parse the cleaned string
                return json.loads(interpretation)
            except json.JSONDecodeError as e:
                # This error should now be less frequent with the better cleanup
                print(f"Error: Failed to decode JSON response AFTER cleaning: {e}")
                print(f"Cleaned interpretation string was:\n---\n{interpretation}\n---")
                # Also print the original raw interpretation for comparison
                print(f"Original raw interpretation was:\n---\n{response.candidates[0].content.parts[0].text}\n---")
                return None
        else:
            print("Error: Interpretation result from LLM was empty after cleaning markdown (if any).")
            print(f"Original raw interpretation was:\n---\n{response.candidates[0].content.parts[0].text}\n---")
            return None

    except Exception as e:
        print(f"An unexpected error occurred during LLM interpretation: {e}")
        import traceback
        traceback.print_exc()
        return None

def digest_prompts(prompt: str) -> str:
    # split_commands = re.split(r'\b(?:and|then|after that|but|or|so)\b', prompt.lower())
    # split_commands = [sentence.strip() for sentence in split_commands]
    # for i in split_commands:
    inter = get_llm_interpretation(prompt)
    print(f"(!) Attempting to Digest: {inter}")
    for action in inter:
        print(f"(!) Digested. Task in queue: {action}")
        executions.execute_plan(action)
# --- digest_prompts function removed as requested ---


# if __name__ == "__main__":
#     print("--- Testing single command ---")
#     # command_to_test = "open up google chrome"
#     command_to_test = input("Command: ") # Another example
#     # command_to_test = "Look up node.js" # The example from the prompt

#     digest_prompts(command_to_test)
