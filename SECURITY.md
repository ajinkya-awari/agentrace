# Security policy

## Scope

AgentTrace handles model credentials, medical-question prompts, trace events, and benchmark artifacts. Treat all provider responses and unpublished results as sensitive until reviewed.

## Never commit

- `GROQ_API_KEY`, `WANDB_API_KEY`, or any other token
- Raw provider prompts or responses
- Restricted datasets or personal medical information
- Unverified benchmark rates, confidence intervals, dashboards, or deployment links

Use notebook or deployment secrets for runtime credentials. If a secret is exposed, revoke it immediately and report the incident privately to the repository owner; do not include the secret in an issue.
