# models/llm_client.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class BlackBoxLLM:
    def __init__(self, model_name: str = "gpt2", max_length: int = 30):
      
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
        """Generates text using the selected HF model.

        This computes a safe number of new tokens to generate based on the
        prompt's token length and the model/tokenizer max position (if
        available). Passing `max_new_tokens` prevents the "input length ==
        max_length" ValueError and avoids exceeding model position limits.
        """
        # tokenize to compute input length
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_len = enc['input_ids'].shape[1]

        # try to get model/tokenizer max position limit
        max_pos = None
        if hasattr(self.model.config, 'n_positions'):
            max_pos = getattr(self.model.config, 'n_positions')
        elif hasattr(self.model.config, 'max_position_embeddings'):
            max_pos = getattr(self.model.config, 'max_position_embeddings')
        else:
            max_pos = getattr(self.tokenizer, 'model_max_length', None)

        # compute allowed new tokens ensuring at least 1 token can be generated
        if isinstance(max_pos, int) and max_pos > input_len:
            allowed_new = max(1, min(self.max_length, max_pos - input_len - 1))
        else:
            allowed_new = max(1, self.max_length)

        outputs = self.generator(
            prompt,
            do_sample=True,
            max_new_tokens=allowed_new
        )
        return outputs[0]['generated_text']
