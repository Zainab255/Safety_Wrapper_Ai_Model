# Safety Wrappers for Black‑Box Language Models

### Purpose 

This project provides a **extensible safety‑wrapper system** that sits around a **black‑box Large Language Model (LLM)** and enforces simple safety policies (**ALLOW, BLOCK, MODIFY, REQUERY**) before returning model outputs to users.

### Problem Addressed 

Large Language Models can sometimes generate **unsafe or harmful content**. Since many production LLMs are **black boxes**, direct modification of their internal behavior is not possible. This project demonstrates how **lightweight, composable safety wrappers** can be applied externally to:

* Detect banned or risky content
* Use recent interaction history
* Control re‑query budgets
* Produce safer final outputs

### Core Idea 

Model outputs are passed through one or more **policy wrappers**. Each wrapper independently evaluates the output and returns a **WrapperDecision** along with an optional payload to control the final response or trigger a re‑query.


##  Quick Start

### Install Dependencies

```bash
pip install transformers torch
```

### Run the Pipeline

```bash
python -m pipeline.runner "your prompt here" [wrapper_type]
```

**Available wrapper types:**

* `baseline`
* `keyword`
* `history`
* `query_budget`

(Default: `keyword`)

### Run Unit Tests

```bash
python run_wrapper_tests.py
```



## 🗂️ Project Structure (High‑Level)

```
project_root/
│
├── pipeline/
│   ├── runner.py          # Orchestrates model calls and wrapper decisions
│   └── logger.py          # JSONL logging utility
│
├── models/
│   └── llm_client.py      # BlackBoxLLM wrapper using Hugging Face pipelines
│
├── wrappers/
│   ├── base.py            # SafetyWrapper base class & WrapperDecision enum
│   ├── keyword_wrapper.py # Blocks outputs with banned keywords
│   ├── history_wrapper.py # Uses recent history to block unsafe outputs
│   ├── query_budget_wrapper.py # Limits number of re‑queries
│   └── noop_wrapper.py    # Baseline wrapper (always ALLOW)
│
├── config/
│   └── config.json        # Central configuration file
|
├── data/                 
│   ├── risky_prompts.jsonl
│   └── harmless_prompts.jsonl
│
├── experiments/          
│   └── run_batch.py       # Script for batch evaluation and metrics
│
├── logs/
│   ├── experiment_001.jsonl
│   └── history.jsonl
│
├── tests/
│   └── run_wrapper_tests.py
│
└── README.md
```



## 🧭 How It Works (Concise Flow)

1. `runner.run_pipeline()` loads conversation history and configuration.

2. A **BlackBoxLLM** generates an initial model output.

3. The selected **wrapper** evaluates the output.

4. Wrapper returns a **WrapperDecision**:

   * **ALLOW** → Output is returned to the user
   * **BLOCK** → A safe refusal message is returned
   * **REQUERY** → Model is queried again using a revised prompt
   * **MODIFY** → Output is altered before returning (optional behavior)

5. Re‑queries continue until a final decision is reached or the query budget is exhausted.

6. All interactions are logged and appended to history.



##  Configuration & Files

All configurable values are centralized in **`config/config.json`**.

### Key Configuration Fields

* `banned_keywords`: List of restricted words
* `safe_refusal`: Message returned when content is blocked
* `log_file`: Path for JSONL experiment logs
* `history_file`: Path for persistent conversation history

### Logging

* Logs are written in **JSONL format** 
* Enables easy experiment analysis and metric computation



##  Extending the Project — Add a New Wrapper

1. Create a new file in `wrappers/`
2. Inherit from `SafetyWrapper`
3. Implement the `decide()` method:

```python
def decide(self, user_prompt: str, model_output: str, history: list):
    return WrapperDecision.ALLOW, None
```

### Return Values

* `WrapperDecision.ALLOW`
* `WrapperDecision.BLOCK`
* `WrapperDecision.MODIFY`
* `WrapperDecision.REQUERY`, with payload:

```python
{"revised_prompt": "..."}
```



##  Testing & Notes

* Wrapper unit tests are provided in `run_wrapper_tests.py`
* Each wrapper is tested independently for expected behavior
* Add new tests when extending functionality


## 📄 License
