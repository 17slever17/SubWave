import os
import sys
import time
import logging

# Ensure the root of the project is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.settings import load_config
from services.llm.translator import LLMTranslator

def run_test_case(translator, name, text, context, style=None):
    print(f"\n>>> Running Test Case: {name}")
    print(f"Context: {context}")
    print(f"Text: {text}")
    if style:
        print(f"Style: {style}")
        # Temporarily override style in config if provided
        original_style = translator.config.system_prompt
        translator.config.system_prompt = style
    
    start_time = time.time()
    result = translator.translate(text, context)
    duration = time.time() - start_time
    
    if style:
        translator.config.system_prompt = original_style
        
    print(f"Result: {result}")
    print(f"Time: {duration:.2f}s")
    return {"name": name, "result": result, "time": duration}

def main():
    # Set logging to ERROR to keep console clean for results
    logging.basicConfig(level=logging.ERROR)
    
    print("Initializing components for full test...")
    config = load_config()
    translator = LLMTranslator(config)
    
    if not translator.llm:
        print("Error: Model failed to load. Check paths and llama-cpp installation.")
        return

    test_cases = [
        {
            "name": "Basic Translation",
            "text": "Hello, how are you today?",
            "context": "",
            "style": "Formal translation."
        },
        {
            "name": "Context Awareness",
            "text": "It was really huge!",
            "context": "I saw a whale in the ocean.",
            "style": "Direct translation."
        },
        {
            "name": "Gen-Z Slang Style",
            "text": "I don't believe what you are saying.",
            "context": "Person A: I just won the lottery!",
            "style": "Translate like an emotional Gen-Z streamer, use modern internet slang, be very expressive."
        },
        {
            "name": "Technical Context",
            "text": "You need to update the drivers.",
            "context": "The user is having issues with their GPU performance in games.",
            "style": "Helpful tech support style."
        }
    ]
    
    results = []
    print("\n" + "="*60)
    print("STARTING INTEGRATION TESTS")
    print("="*60)
    
    for case in test_cases:
        res = run_test_case(translator, case["name"], case["text"], case["context"], case["style"])
        results.append(res)
        
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    for res in results:
        # Truncate result for summary
        short_res = (res["result"][:50] + '...') if len(res["result"]) > 50 else res["result"]
        print(f"[{res['time']:.2f}s] {res['name']}: {short_res}")
    print("="*60)

if __name__ == "__main__":
    main()
