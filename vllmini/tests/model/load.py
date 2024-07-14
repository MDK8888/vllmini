from transformers import GPT2Config
from vllmini.model.gpt2 import GPT2LMHeadModel

def test_weight_loading():
    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    gpt2_model = GPT2LMHeadModel(config)
    gpt2_model.load_huggingface_weights(model_name)

if __name__ == "__main__":
    test_weight_loading()

