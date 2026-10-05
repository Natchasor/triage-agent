# Support Ticket Triage Agent

An AI agent that triages incoming customer-support tickets. For each ticket it:

1. **Classifies urgency**: critical, high, medium or low
2. **Extracts** the product, issue type, customer sentiment and language
3. **Searches a knowledge base** for relevant help articles
4. **Decides the next action**: auto-respond, route to a specialist team, or escalate to a human
5. **Drafts a reply** in the customer's own language

It runs in the terminal, uses the OpenAI API, and ships with three sample tickets (one is in
Thai). A one-page write-up of the design is in [WRITEUP.md](WRITEUP.md) (PDF copy:
[WRITEUP.pdf](WRITEUP.pdf)).

## Quick start

Requirements: **Python 3.11 or newer** and an **OpenAI API key**.

**1. Get the code and install it**

```bash
git clone <your-repo-url> triage-agent
cd triage-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

**2. Add your API key**

```bash
cp .env.example .env
```

Then open `.env` and replace `your-key-here` with your OpenAI API key. On Windows, use
`copy .env.example .env` instead.

**3. Run it**

```bash
python -m triage.cli
```

This triages the three sample tickets. Use `python -m ...` for every command in this file, so the
tools from this virtual environment are the ones that run.

## Configuration

Settings are read from `.env` (copied from `.env.example`):

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | none | Your OpenAI API key |
| `OPENAI_MODEL` | no | `gpt-5.6-terra` | Any model that supports the Responses API with function calling |
| `OPENAI_REASONING_EFFORT` | no | model default | For example `low`, to reduce latency |

To check your key and model before running the agent:

```bash
python scripts/check_api.py
```

It should print the model name followed by `ready`.

## Usage

```bash
python -m triage.cli
python -m triage.cli --ticket T-1002
python -m triage.cli --json
python -m triage.cli -v
```

| Command | What it does |
|---|---|
| `python -m triage.cli` | Triage all the sample tickets |
| `--ticket T-1002` | Triage one ticket |
| `--json` | Print the full result as JSON |
| `-v` | Also log every tool call |

Each ticket prints its urgency, action (and team), extracted fields, the knowledge-base articles
used, the tools called, any guardrail overrides, and a draft reply. Nothing is actually sent to
customers.

Example output (abbreviated):

```
=== T-1001 ==================================================
Urgency:     high
Action:      escalate_human -> billing
Issue:       billing | Pro plan upgrade and Pro export access
Sentiment:   angry
Language:    en
Confidence:  0.98
KB articles: kb_002, kb_001, kb_008
Tools used:  get_customer_history, get_billing_transactions, search_knowledge_base
Guardrails:  none triggered
Draft reply:
    I'm sorry for the delay. I can confirm that there are three pending $29.99 Pro-upgrade
    transactions on your account, while your plan is still showing as Free. I've escalated
    this to our Billing team for urgent verification ...
```

The sample tickets are in `data/tickets.json`:

- **T-1001**: failed upgrade, three charges, chargeback threat (billing)
- **T-1002**: Thai-language outage report from an Enterprise customer; the public status page
  says everything is fine
- **T-1003**: dark-mode question plus a feature request

## Tests

```bash
python -m pytest
python -m ruff check .
```

The tests need no API key: they run the agent against a scripted fake OpenAI client.

## How it works

```
ticket -> TriageAgent (LLM + tools loop) -> decision -> guardrails -> final result
```

1. The agent sends the ticket and the system prompt to the model.
2. The model calls tools to gather facts. Their results go back to the model.
3. The model finishes by calling `submit_triage_decision`. The output is validated against the
   `TriageDecision` schema; invalid output is sent back to the model to fix.
4. Deterministic guardrails then review the decision. They can only make it more cautious
   (for example, turning an auto-reply into an escalation). Every override is logged.

| What | Where |
|---|---|
| System prompt | `src/triage/prompts.py` |
| Tool definitions (schema and implementation) | `src/triage/tools/` |
| Agent loop | `src/triage/agent.py` |
| Guardrails | `src/triage/guardrails.py` |
| Output schema and domain models | `src/triage/models.py` |
| Mock data (tickets, customers, knowledge base, billing, status) | `data/` |

The four tools, all backed by the mock data in `data/`:

| Tool | Purpose |
|---|---|
| `get_customer_history` | Plan, region, seats, tenure, previous tickets |
| `search_knowledge_base` | Keyword search over the FAQ and help articles |
| `get_billing_transactions` | Recent payments, with totals and duplicate detection computed in code |
| `check_system_status` | Internal region health compared with the public status page |

## Extending

- **Add a tool:** create a module in `src/triage/tools/` with a class that sets `name`,
  `description` and `args_model` and implements `run()`. Then add one line to
  `src/triage/tools/__init__.py`. The schema sent to the model is generated from `args_model`.
- **Add a guardrail:** write a function `(decision, context) -> Override | None` in
  `src/triage/guardrails.py` and add it to `DEFAULT_RULES`.

## Troubleshooting

- **"Cannot create the OpenAI client"**: `OPENAI_API_KEY` is missing. Check your `.env` file.
- **`model_not_found` or a 404 error**: your key can't use the default model. Set `OPENAI_MODEL`
  in `.env` to another model that supports the Responses API.
- **`ModuleNotFoundError: triage`**: the virtual environment isn't active. Activate it and try
  again.

## Known limitations

- Tool back-ends are mocked with JSON files, and knowledge-base search is weighted keyword
  matching (a production system would use embeddings).
- Draft replies are printed, never sent.
- Model output varies slightly between runs on borderline labels such as sentiment.
- Verified with the three sample tickets. WRITEUP.md describes how I would evaluate it properly.
