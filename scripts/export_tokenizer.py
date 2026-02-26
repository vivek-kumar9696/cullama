import struct
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "tokenizer.bin"

def export_tokenizer():
    print(f"Loading Tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = tokenizer.vocab_size

    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        
        for i in range(vocab_size):
            # Get the raw token string
            token = tokenizer.convert_ids_to_tokens(i)
            if token is None:
                token = ""
                
            # SentencePiece uses " " (U+2581) to represent a space. 
            # We replace it with an actual space for C++ to print properly.
            token = token.replace(' ', ' ')
            
            # Handle the literal newline token
            if token == "<0x0A>":
                token = "\n"
            
            token_bytes = token.encode('utf-8')
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

    print(f"[Success] Vocabulary exported to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_tokenizer()