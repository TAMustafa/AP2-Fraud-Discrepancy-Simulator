# AP2 Fraud/Discrepancy Simulator

A small, educational simulator that demonstrates AP2-style agent payment flows using signed mandates and scenario-based validation.

The simulator models a basic pipeline:

- User creates an Intent mandate and signs it.
- Agent creates a Cart mandate from the Intent and signs it.
- Merchant creates a Payment mandate from the Cart and signs it.
- A Credentials step can optionally tamper with the signature to simulate failures.

You can run the flow deterministically (no LLM) or toggle an optional LLM-driven agent (OpenAI via LangChain) to choose item/price suggestions.

## Features

- HMAC-SHA256 signing over canonical JSON (demo-grade; not for production).
- Structural validation of parent-child mandate linkage.
- Business rule validation (amount caps, item/category alignment, currency, expiry).
- Signature verification for each mandate.
- Scenarios to demonstrate discrepancies: Bad Agent, Bad Merchant, Expired Mandate, Invalid Signature.
- Optional LLM agent using OpenAI through `langchain-openai` (off by default).

## Project Structure

- `app.py` – main simulator and Gradio UI.
- `pyproject.toml` – dependencies and Python constraints.
- `README.md` – this file.

## Requirements

- Python 3.12+
- Dependencies declared in `pyproject.toml`:
  - `gradio`
  - `python-dotenv`
  - `langchain`, `langchain-community` (core plumbing)
  - `langchain-openai` (optional; only needed for LLM agent mode)

## Install

Using uv (recommended):

```bash
# from repository root
uv sync
```

Using pip/venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Optional: Configure OpenAI for the LLM agent

If you want to enable the LLM-driven agent, set an OpenAI key. You can use a `.env` file at the repo root or export as env variables.

`.env` example:

```
OPENAI_API_KEY=sk-...
# Optional model override
OPENAI_MODEL=gpt-4o-mini
```

Without `OPENAI_API_KEY`, the app continues to work in deterministic mode.

## Run

```bash
# uv
uv run python app.py

# OR pip/venv
python app.py
```

Open the Gradio UI URL printed in the console.

- Choose a scenario in the dropdown.
- Optionally check "Use LLM Agent (OpenAI)" to let the agent pick item/price suggestions via LLM.
- The left output shows the signed mandates (`intent`, `cart`, `payment`).
- The right output shows validation results and discrepancies.

## What the Simulator Does

- Builds mandates with IDs and parent references to mirror AP2’s flow semantics.
- Applies HMAC signatures over a canonical JSON representation to demonstrate integrity checks.
- Validates:
  - Parent references (Intent → Cart → Payment).
  - Price within `max_amount`.
  - Item/category alignment (Agent shouldn’t deviate from requested category).
  - Amount and currency consistency.
  - Expiration of the Intent.
  - Signatures for each mandate.
- Provides scenario toggles that create realistic discrepancy cases:
  - Bad Agent: agent inflates price beyond `max_amount`.
  - Bad Merchant: merchant swaps the item.
  - Expired Mandate: user’s intent is expired on creation.
  - Invalid Signature: credentials step breaks the payment signature.

## Alignment with AP2 Recommendations

- Clear data boundaries and references between mandates (`mandate_id`, `parent_id`).
- Signing and verification of mandates across roles.
- Scenario-first thinking to test resilience of downstream validation.
- Optional agentic behavior with an LLM (while keeping the deterministic path as default).

Note: This project uses demo-grade signing (HMAC over canonical JSON). For production or closer alignment with real AP2 deployments:

- Use established cryptographic libraries and public/private key signatures.
- Include explicit signature metadata (algorithm, key id) and robust canonicalization.
- Consider adopting AP2’s types package when it is published to PyPI or installable via git.

## Environment Variables

- `USER_SECRET`, `AGENT_SECRET`, `MERCHANT_SECRET`, `CREDENTIALS_SECRET` – shared secrets for HMAC signing (defaults are provided for demos).
- `OPENAI_API_KEY` – enables LLM agent mode; if missing, the simulator runs deterministically.
- `OPENAI_MODEL` – optional model name override (default: `gpt-4o-mini`).

## Troubleshooting

- Missing module: Make sure you’ve installed dependencies via `uv sync` or `pip install -e .`.
- No LLM: Ensure `OPENAI_API_KEY` is set and `langchain-openai` is installed (it is declared in `pyproject.toml`).
- Port conflicts: Gradio defaults may conflict if multiple apps are running. Adjust host/port in `demo.launch()` as needed.

## Deploying to Hugging Face Spaces

You can host this app on Hugging Face Spaces with a stable public URL.

Steps:

1. __Create a new Space__
   - Go to https://huggingface.co/new-space
   - Choose SDK: Gradio
   - Runtime: Python 3.10+ (3.12 is fine)

2. __Upload repo files__
   - Include at minimum: `app.py`, `requirements.txt`, and this `README.md`.
   - The app defines a `demo = gr.Interface(...)` object in `app.py`. On Spaces, the platform serves this automatically. In local mode only, `demo.launch()` is called.

3. __Configure secrets__ (if using OpenAI LLM agent)
   - In the Space Settings → Secrets, add:
     - `OPENAI_API_KEY`: your API key
     - Optional: `OPENAI_MODEL` (defaults to `gpt-4o-mini`)
   - Do not commit `.env` to the Space. The app uses environment variables directly on Spaces.

4. __Dependencies__
   - Spaces installs from `requirements.txt` automatically.

5. __Run__
   - The Space will build and start the app. You’ll get a public URL when it’s live.

Notes:
- The Flag button is disabled (`allow_flagging="never"`).
- The LLM Agent checkbox is optional; leave it off to run deterministically.

## License

This simulator is for demonstration and educational purposes.
