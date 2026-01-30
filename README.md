# Safety Wrapper for Black-box LLMs

Confidential research project for Zia Ush Shamszaman (Teesside University)

Goal: build external safety wrappers around a black-box LLM and measure safety/usefulness trade-offs.

Quick start

1. Create virtual environment and activate:
   - python -m venv venv
   - venv\Scripts\activate
2. Install requirements:
   - pip install -r requirements.txt
3. Copy .env.example to .env and add API keys (optional for OpenAI)
4. Run an example batch:
   - python scripts/run_batch.py --prompts data/prompts_benign.jsonl --wrapper wrappers.keyword.KeywordWrapper

Structure

- wrappers/  - wrapper implementations and base classes
- models/    - model adapters (OpenAI / HF)
- pipeline/  - runner and logging pipeline
- data/      - prompt sets (JSONL/CSV)
- logs/      - output logs (JSONL)
- experiments/ - analysis outputs and plots
- docs/      - design notes and templates

Contact: Zia Ush Shamszaman

License: Confidential (internal research)
