import gptmodel

def main():

    GPT_CONFIG_124M = {
            "vocab_size": 50257,    #Vocabulary size
            "context_length": 1024, #Context length
            "emb_dim": 768,         #Embedding dimensions
            "n_heads": 12,          #Number of attention heads
            "n_layers": 12,         #Number of layers
            "drop_rate": 0.1        #Dropout rate
            "qkv_bias": False       #Query-Key-Value bias

    gpp_model = DummyGPTModel(GPT_CONFIG_124M)

if __name__ == "__main__":
    main()
