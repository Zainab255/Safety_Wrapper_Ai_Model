import json
import pandas as pd
from pipeline.runner import run_pipeline

# Choose which dataset to run
DATASET = "data/risky_prompts.jsonl"  # change to harmless_prompts.jsonl
LOG_FILE = "logs/experiment_batch.jsonl"


def load_prompts(file_path):
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def main():
    prompts = load_prompts(DATASET)

    results = []
    for prompt_entry in prompts:
        user_prompt = prompt_entry["text"]
        final_output = run_pipeline(user_prompt)

        results.append({
            "user_prompt": user_prompt,
            "category": prompt_entry["category"],
            "final_output": final_output
        })

    # Save summary CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv("logs/batch_results.csv", index=False)

    print("Batch run complete. Results saved to logs/batch_results.csv")


if __name__ == "__main__":
    main()
