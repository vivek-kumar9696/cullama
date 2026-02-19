import torch
import struct
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "model.bin"

def export_model():
    print(f"[Exporter] Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("cpu")
    config = model.config

    print("[Exporter] Processing weights...")
    
    with open(OUTPUT_FILE, "wb") as f:
        # 1. Write Header (Integers)
        # Magic Number (0xLLAMA)
        f.write(struct.pack("i", 0x4C4C414D)) # ASCII for "LLAM" 
        # Model Hyperparameters
        f.write(struct.pack("i", config.hidden_size))     # dim
        f.write(struct.pack("i", config.intermediate_size)) # hidden_dim (SwiGLU)
        f.write(struct.pack("i", config.num_hidden_layers)) # n_layers
        f.write(struct.pack("i", config.num_attention_heads)) # n_heads
        f.write(struct.pack("i", config.num_key_value_heads)) # n_kv_heads
        f.write(struct.pack("i", config.vocab_size))
        f.write(struct.pack("i", 2048)) # Max Seq Len (Hardcoded for TinyLlama)

        print(f"Header: Dim={config.hidden_size}, Layers={config.num_hidden_layers}")

        # 2. Iterate and Write Weights in Order
        # We need a strict order for C++ to read them blindly.
        state_dict = model.state_dict()
        
        # Helper to write a tensor
        def write_tensor(name):
            if name not in state_dict:
                print(f"ERROR: Missing {name}")
                return
            t = state_dict[name].float().numpy()
            f.write(t.tobytes())
            print(f"  -> Wrote {name} ({t.shape})")

        # A. Embeddings
        write_tensor("model.embed_tokens.weight")

        # B. Layers
        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            # Attention
            write_tensor(f"{prefix}.input_layernorm.weight") # RMSNorm
            write_tensor(f"{prefix}.self_attn.q_proj.weight")
            write_tensor(f"{prefix}.self_attn.k_proj.weight")
            write_tensor(f"{prefix}.self_attn.v_proj.weight")
            write_tensor(f"{prefix}.self_attn.o_proj.weight")
            # MLP (SwiGLU)
            write_tensor(f"{prefix}.post_attention_layernorm.weight") # RMSNorm
            write_tensor(f"{prefix}.mlp.gate_proj.weight")
            write_tensor(f"{prefix}.mlp.up_proj.weight")
            write_tensor(f"{prefix}.mlp.down_proj.weight")

        # C. Final Norm & Head
        write_tensor("model.norm.weight")
        write_tensor("lm_head.weight")

    print(f"[Success] Model saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_model()