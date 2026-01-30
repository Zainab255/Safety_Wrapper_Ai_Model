# models/llm_client.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class BlackBoxLLM:
    def __init__(self, model_name: str = "gpt2", max_length: int = 25):
      
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(self, prompt: str):
        """Generates text using the selected HF model."""
        outputs = self.generator(prompt, do_sample=True, min_length=self.max_length, max_length=self.max_length)
        return outputs[0]['generated_text']
