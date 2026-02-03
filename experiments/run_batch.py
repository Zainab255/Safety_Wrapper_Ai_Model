import pandas as pd
import json
import os
from pipeline.runner import run_pipeline, load_config
from models.llm_client import BlackBoxLLM

CONFIG = load_config()

def run_all_experiments():
    models = CONFIG["models"]
    wrappers = ["baseline", "keyword", "history", "query_budget"]
    datasets = ["data/risky_prompts.jsonl", "data/harmless_prompts.jsonl"]

    # Iterate over every Model
    for model_name in models:
        print(f"\n==========================================")
        print(f"LOADING MODEL: {model_name}")
        print(f"==========================================")
        
        try:
            # Initialize model once per session
            llm = BlackBoxLLM(model_name, hf_token=CONFIG["hf_token"])
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")
            continue

        # Iterate over Data
        for dataset_path in datasets:
            data_name = "risky" if "risky" in dataset_path else "harmless"
            
            # Load Prompts
            prompts = []
            with open(dataset_path, "r") as f:
                for line in f:
                    prompts.append(json.loads(line)["text"])

            # Iterate over Wrappers
            for wrapper in wrappers:
                print(f"--> Running: Model={model_name} | Wrapper={wrapper} | Data={data_name}")
                
                results = []
                total_calls = 0
                blocked_count = 0

                for i, p in enumerate(prompts):
                    # Run Pipeline
                    final_out, calls = run_pipeline(p, wrapper, llm, CONFIG["history_file"])
                    
                    # Metrics
                    total_calls += calls
                    is_blocked = (final_out == CONFIG["safe_refusal"])
                    if is_blocked: blocked_count += 1
                    
                    results.append({
                        "prompt": p,
                        "output": final_out,
                        "calls": calls,
                        "blocked": is_blocked
                    })

                # Save Results to CSV
                df = pd.DataFrame(results)
                clean_model_name = model_name.split("/")[-1]
                filename = f"logs/results_{clean_model_name}_{wrapper}_{data_name}.csv"
                df.to_csv(filename, index=False)
                
                # Print Summary
                avg_calls = round(total_calls / len(prompts), 2)
                block_rate = round(blocked_count / len(prompts), 2)
                print(f"    [Done] Saved to {filename}")
                print(f"    Metrics: Block Rate={block_rate}, Avg Calls={avg_calls}")

if __name__ == "__main__":
    run_all_experiments()