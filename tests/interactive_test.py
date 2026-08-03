import os
import sys
import time

# Get the directory of the current file (tests)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to get to the project root (Youtube_Codding)
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.settings import load_config
from services.llm.translator import LLMTranslator

def main():
    print("Loading configuration...")
    config = load_config()
    
    print("Initializing LLM Translator (this may take a moment)...")
    translator = LLMTranslator(config)
    
    print("\n" + "="*50)
    print("LLM Translator Interactive Test")
    print("="*50)
    print("Type your text and press Enter to translate.")
    print("Type 'exit' or 'quit' to quit.")
    print("="*50 + "\n")
    
    previous_context = ""
    
    while True:
        try:
            target_text = input("Enter text: ").strip()
            
            if target_text.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
                
            if not target_text:
                continue
                
            print("\nGenerating translation...")
            start_time = time.time()
            
            translated_text = translator.translate(target_text, previous_context)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"\n[Translation]: {translated_text}")
            print(f"[Time]: {generation_time:.2f} seconds\n")
            
            # Update context for the next translation
            previous_context = target_text
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()
