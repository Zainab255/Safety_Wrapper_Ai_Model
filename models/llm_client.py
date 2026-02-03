import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class BlackBoxLLM:
    def __init__(self, model_name: str, hf_token: str = None):
        print(f"Loading Model: {model_name}...")
        self.model_name = model_name
        
        # Detect device (GPU if available, else CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                token=hf_token,
                device_map="auto" if self.device == 0 else None,
                torch_dtype="auto"
            )
            
            # Create pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                # device=self.device # pipeline handles device_map automatically usually
            )
        except Exception as e:
            print(f"FAILED to load {model_name}. Error: {e}")
            raise e

    def generate(self, prompt: str):
        """Generates text using the selected HF model."""
        try:
            # Generate with some randomness (temperature) to test safety variablity
            outputs = self.generator(
                prompt, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # Extract just the new text, removing the prompt
            full_text = outputs[0]['generated_text']
            # Simple cleanup to return only the answer if possible
            return full_text[len(prompt):].strip()
        except Exception as e:
            return f"[Model Error: {str(e)}]"