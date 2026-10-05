# Write-up: Support Ticket Triage Agent

## Architecture decisions

- **Tool-calling agent loop, no framework.** The model chooses among four tools (customer history, knowledge-base search, billing, system status); a ~100-line loop runs them and feeds results back. I rejected a single prompt (not really an agent), a multi-step pipeline (more cost and latency for no gain here) and a framework (hides the logic being judged).
- **The final answer is a tool call.** `submit_triage_decision` takes the `TriageDecision` schema. Invalid output is rejected with the validation error so the model can fix it; the last round forces submission, and a round cap bounds cost.
- **Responses API.** My first version used Chat Completions, which OpenAI rejected for function tools combined with reasoning on the chosen model. The model name is an environment variable, so it works with the reviewer's key and model.
- **The LLM proposes, code disposes.** Seven deterministic guardrails run after the model: critical, dispute or legal wording, and Enterprise plus high urgency all go to a human; angry customers are never auto-answered; confidence below 0.6 goes to a human; an auto-reply must cite a real knowledge-base article; team and action must agree. They can only make a decision more cautious (property-tested) and rely on our own data, not the model's claims.
- **Code does the facts and arithmetic.** Billing totals, duplicate-charge detection and "status page contradicts internal health" are computed in the tools. Tool errors return as data, so the model recovers: it once queried "Asia", got the valid regions back, and retried with "TH".
- **Testable offline.** The OpenAI client is injected, so 68 tests run against a scripted fake with no API key. Tool back-ends are JSON files and KB search is weighted keywords; each is replaceable in one file.

## What could go wrong, and how I would handle it

- **Wrong urgency or action:** the prompt rates urgency by business impact, not tone; guardrails and low-confidence escalation catch the worst misses. Under-escalation is the costly error, so every guardrail pushes toward a human.
- **Invented answers or promises:** replies may state only tool or KB facts and never promise refunds or timelines. Gap: the guardrail proves the cited article exists, not that it supports the reply.
- **Prompt injection:** ticket text is fenced as data; tools are read-only; output is schema-validated; guardrails do not trust the model.
- **Tool or API failure:** errors return to the model; requests have timeouts and retries; one failed ticket does not stop the rest. Production would queue failures for a human.
- **Drift and variance:** pin the model. Borderline labels varied between runs (T-1003's sentiment flipped) while urgency and action stayed stable.
- **Replies that claim actions:** "we have recorded your request" is only true if something records it. Production needs a logging tool.
- **Privacy:** ticket text goes to a third-party API; production needs PII redaction and a data-processing agreement.

## How I would evaluate it in production

- **Offline, on every prompt or model change:** a labelled golden set of about 200 tickets (multilingual, adversarial, borderline). Track urgency accuracy (adjacent levels tolerated), action accuracy, and above all the **under-escalation rate**: tickets needing a human that were auto-answered. Check replies for language match, no promises, and claims supported by the cited article (LLM judge plus human sampling). Repeat each ticket several times to measure consistency.
- **Rollout:** shadow mode first (agent suggests, humans decide), then auto-respond only for low-risk categories with sampled audits.
- **Online:** human override rate, guardrail trigger rate (a rise signals drift), escalation rate, reopen rate and CSAT for auto-answered tickets, time to first response, and cost and tool rounds per ticket (p95).
