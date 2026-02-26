import subprocess
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("[Frontend] Loading HuggingFace Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # The perfectly formatted Chat Prompt
    user_input = "What is the capital of France?"
    
    chat_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
    
    # Convert string to Token IDs
    tokens = tokenizer.encode(chat_prompt)
    
    print(f"[Frontend] Prompt tokenized into {len(tokens)} tokens.")
    print("[Frontend] Passing control to Bare-Metal C++ Engine...\n")
    
    # Build the command: ./build/cuLlama model.bin tokenizer.bin 1 345 23 ...
    command =["./build/cuLlama", "model.bin", "tokenizer.bin"] + [str(t) for t in tokens]
    
    # Execute the C++ Engine!
    subprocess.run(command)

if __name__ == "__main__":
    main()