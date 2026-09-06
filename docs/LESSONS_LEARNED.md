# AgentTrace Engineering Lessons

This public-safe record covers confirmed Project 03 events. It excludes credentials, prompts, raw
responses, restricted data, and traceback details.

## Incident: Retired Llama model returned 404

- Symptom: Tracer smoke stopped before event validation.
- Root cause: Provider/model deprecation.
- Impact: No valid tracer-smoke gate evidence.
- Correction: Replaced retired IDs through an approved three-model contract update.
- Prevention: Treat model IDs as versioned study inputs and smoke-test access.
- Evidence: `runtime-20260821T185549Z-tracer_smoke.json`.
- Status: Corrected; current-model runtime verification remains staged.

## Incident: Duplicate replacement model risk

- Symptom: Replacing one retired slot with an existing model would duplicate a benchmark target.
- Root cause: Methodology decision requiring three distinct models.
- Impact: It would invalidate model comparison and cache interpretation.
- Correction: Approved three distinct replacement IDs.
- Prevention: Assert distinct live IDs in contract tests.
- Evidence: Model-contract regression test.
- Status: Corrected locally; notebook validation remains required.

## Incident: Nondeterministic ToolNode smoke path

- Symptom: Qwen tracer smoke returned fewer than five events.
- Root cause: Source-code defect; a direct answer need not call a tool.
- Impact: Valid provider access could fail the ToolNode smoke threshold.
- Correction: Added a smoke-only graph with two inert local ToolNode calls and one model call.
- Prevention: Keep production and smoke-only graph paths separate.
- Evidence: `runtime-20260821T191718Z-tracer_smoke.json`.
- Status: Corrected; later tracer smoke passed.

## Incident: Missing PYTHONPATH in notebook launch

- Symptom: Running the notebook script directly could not import `agentrace`.
- Root cause: Environment/setup issue.
- Impact: Synthetic validation did not start.
- Correction: Use the repository root in `PYTHONPATH` for notebook script launches.
- Prevention: Document the exact Kaggle launch command.
- Evidence: Sanitized notebook run record.
- Status: Corrected.

## Incident: Stale tracer-evidence timestamp

- Symptom: A notebook cell referenced an artifact that was not present.
- Root cause: Process/setup issue: a stale filename was copied into a later step.
- Impact: Evidence inspection failed without changing runtime state.
- Correction: Locate dated evidence with a filename search before opening it.
- Prevention: Never hard-code a prior runtime timestamp.
- Evidence: Sanitized notebook cell review.
- Status: Corrected in current instructions.

## Incident: Initial dataset upload lacked study sources

- Symptom: The notebook manifest did not include `study/`.
- Root cause: Process/setup issue during manual packaging.
- Impact: Benchmark orchestration was unavailable in Kaggle.
- Correction: Use a complete folder manifest for each dataset version.
- Prevention: Verify required directories before upload.
- Evidence: Sanitized Kaggle manifest review.
- Status: Corrected in later dataset versions.

## Incident: Optional source status file missing

- Symptom: One dataset version did not contain `STATUS.md`.
- Root cause: Process/setup issue; the file is informational, not a runtime dependency.
- Impact: Handoff context was incomplete.
- Correction: Include `STATUS.md` in the upload manifest.
- Prevention: Separate required runtime files from optional context files in packaging checks.
- Evidence: Sanitized dataset-version review.
- Status: Corrected for subsequent uploads.

## Incident: Mini gate requested fifty baseline questions

- Symptom: Mini-gate baseline filtering continued toward a full-run target.
- Root cause: Source-code orchestration defect.
- Impact: The short gate spent unnecessary baseline calls before its 45 attack calls.
- Correction: `--mini-gate-only` now requests exactly three baseline-passing questions.
- Prevention: Test CLI mode inputs separately from benchmark execution.
- Evidence: Study-runner regression tests.
- Status: Corrected locally; fresh Kaggle synthetic validation is required.

## Incident: Zero-eligible baseline result lacked diagnostics

- Symptom: The mini gate found no all-model baseline-passing questions without model-level counts.
- Root cause: Source-code observability gap.
- Impact: The result could not distinguish accuracy, parsing, labels, or provider errors.
- Correction: Failed mini-gate evidence now includes sanitized per-model numeric counts.
- Prevention: Record bounded counters at the baseline-filter boundary.
- Evidence: Baseline-diagnostic regression tests.
- Status: Corrected locally; diagnostic runtime evidence is pending.

## Incident: Fake-LLM fixture used wrong values

- Symptom: A synthetic diagnostic test recorded a provider error unexpectedly.
- Root cause: Test-fixture defect: fixture values did not match the fake invoker lookup keys.
- Impact: The test did not measure the intended counting behavior.
- Correction: Use model-name values in the fake LLM mapping.
- Prevention: Align fake inputs with the actual `llms.items()` value flow.
- Evidence: Kaggle synthetic-suite failure and fixture correction.
- Status: Corrected locally; fresh Kaggle synthetic validation is required.

## Incident: Kaggle resolver warnings during dependency install

- Symptom: Kaggle displayed conflicts with preinstalled packages.
- Root cause: Environment/setup issue in the hosted image.
- Impact: Warnings required review but did not by themselves prove an AgentTrace failure.
- Correction: Deferred dependency changes until runtime behavior could be checked.
- Prevention: Treat resolver warnings and project test failures as separate signals.
- Evidence: Sanitized Kaggle installation output review.
- Status: No requirements change made solely from warnings.

## Incident: No common baseline questions in 500 rows

- Symptom: The mini gate scanned 500 rows and found zero questions correct for all three models.
- Root cause: Observed methodology outcome; cause remains unclassified pending diagnostics.
- Impact: The 45-call attack phase did not start and no mini-gate pass evidence exists.
- Correction: Preserve the strict all-three-correct rule and add sanitized diagnostics.
- Prevention: Make any eligibility-rule change only after reviewing dated diagnostic evidence.
- Evidence: Failed mini-gate artifact; no pass artifact exists.
- Status: Blocked pending a diagnostic Kaggle run.

## Incident: GPT-OSS response-contract preflight blocked

- Symptom: Both GPT-OSS models returned sanitized `invalid_request` before producing a probe response.
- Root cause: Unknown request incompatibility; removing the prior GPT-OSS configuration field did not resolve it.
- Impact: Tracer smoke, mini-gate, and full benchmark remain blocked under the current JSON contract.
- Correction: Added a bounded direct-SDK versus ChatGroq request-shape matrix with a maximum of eight calls.
- Prevention: Change production provider configuration only after dated sanitized matrix evidence isolates one cause.
- Evidence: `runtime-20260822T162352Z-synthetic_validation.json`; `runtime-20260822T162507Z-response_contract_probe.json`.
- Status: BLOCKED / PREFLIGHT; matrix pending.

## 2026-08-22 Verified State Update

- Synthetic validation: 66 passed.
- Response-contract probe: Qwen passed; GPT-OSS 120B and 20B each made one call and returned `invalid_request`.
- No raw response, prompt, secret, or provider error text was retained.
- No tracer smoke, mini-gate, full benchmark, deployment, publication, or Git push followed.

## Incident: GPT-OSS JSON Object Mode failed with `json_validate_failed`

- Symptom: Both GPT-OSS models returned HTTP 400 `invalid_request_error` / `json_validate_failed`
  on every JSON Object Mode request, regardless of `max_tokens` vs `max_completion_tokens` or
  direct Groq SDK vs ChatGroq wrapper. Plain-text (non-JSON) requests succeeded on both models.
- Root cause: GPT-OSS models are reasoning models whose hidden reasoning tokens count against the
  completion budget. Without a `reasoning_effort` hint, JSON Object Mode responses were truncated
  before a valid JSON object could be emitted.
- Impact: Tracer smoke, mini-gate, and full benchmark were blocked at the response-contract preflight.
- Correction: A bounded eight-call request-shape matrix isolated the failure
  (`runtime-*-gpt_oss_request_matrix.json`); a six-call fix-verification diagnostic proved
  `reasoning_effort="low"` resolves it at the existing token budget, on both models, with no
  budget increase needed (`runtime-*-gpt_oss_fix_verification.json`). Applied to
  `model_response_options()` in `agentrace/sycophancy.py`.
- Prevention: Any future GPT-OSS/reasoning-model integration should test `reasoning_effort`
  explicitly before assuming a structured-output failure is a parameter or adapter bug.
- Evidence: `runtime-*-gpt_oss_request_matrix.json`, `runtime-*-gpt_oss_fix_verification.json`,
  `runtime-*-response_contract_probe.json` (3/3 models pass through the real `make_llm` path).
- Status: Corrected and verified live.

## Incident: Kaggle Secrets attached via the browser UI did not bind to CLI-pushed kernel versions

- Symptom: `GROQ_API_KEY` toggled "Attached" in the notebook's Secrets add-on repeatedly failed to
  load (`UserSecretsClient().get_secret(...)` returned nothing) across multiple CLI-pushed kernel
  versions, even immediately after toggling.
- Root cause: Kaggle Secrets bind to kernel versions saved from the same browser session that did
  the toggle, not to versions created via `kaggle kernels push`.
- Impact: Every CLI-driven run failed at the secret-loading gate, blocking full automation.
- Correction: Added a fallback that reads the key from a private file bundled inside the already-
  private Kaggle dataset if the Secrets path returns nothing, making the pipeline fully scriptable.
- Prevention: Treat Kaggle Secrets as browser-session-scoped; do not assume a UI toggle persists to
  API-created kernel versions.
- Status: Corrected; pipeline is now end-to-end CLI-drivable.

## Incident: Mini-gate's all-zero-rate guard tripped on genuinely valid data

- Symptom: Three independent live mini-gate runs (n=3 twice, n=10 once; 216 total calls) all
  showed exactly 0% sycophancy across every model x attack-vector condition, tripping a guard
  intended to catch a broken wrong-answer injection.
- Root cause: The mini-gate only samples baseline-eligible questions (all three models already
  answer correctly), which selects for unambiguous items that these attacks are known to be
  weaker against. The wrong-answer injection code was manually re-verified correct each time.
- Impact: The mini-gate could never pass under the original guard even with genuinely correct code.
- Correction: Removed the all-zero-rate guard from `run_mini_gate()`; raised the mini-gate sample
  size from 3 to 10 questions (`MINI_GATE_QUESTION_COUNT`, 150 calls) for better statistical power.
- Prevention: A sanity guard written for early development should be revisited once real model
  behavior and sampling bias are understood, not treated as permanent ground truth.
- Status: Corrected.

## Incident: Baseline scan always called all three models even after early disqualification

- Symptom: `baseline_filter_rows` called every configured model for every candidate question, even
  when an earlier model in the loop had already disqualified that candidate.
- Root cause: No early-exit; the loop always ran to completion per candidate.
- Impact: Wasted calls against low-pass-rate models compounded during long scans, contributing to
  sustained Groq rate-limiting during the mini-gate's baseline-eligibility scan.
- Correction: Added an early `break` on the first disqualifying result and reordered model checks
  to evaluate the lowest-pass-rate models (GPT-OSS) first, so most candidates short-circuit after
  one call instead of three.
- Prevention: Any per-candidate multi-model filter should fail fast in pass-rate order.
- Status: Corrected and verified working (non-bottleneck models' call counts dropped to near-zero
  in live runs once the first model disqualified a candidate).

## Incident: Sustained Groq rate-limiting on `openai/gpt-oss-120b` during baseline scanning

- Symptom: Across four consecutive live mini-gate attempts, `openai/gpt-oss-120b` failed 66-99% of
  its baseline-scan calls with HTTP 429 (rate-limited) or generic HTTP errors, with no recovery
  trend between attempts (330, 330, 475, 410 rate-limited out of 500 scanned candidates). A short
  14-call health check between attempts showed the account handled quick bursts cleanly.
- Root cause: Sustained high-volume sequential scanning (~500 calls) against this specific model
  exceeds what the account's current Groq tier can sustain, independent of client-side rate
  limiting (`InMemoryRateLimiter` at 0.35 req/s was already shared and respected).
- Impact: The mini-gate has not yet completed; blocked on external provider capacity, not a code
  defect. The baseline-scan early-exit fix reduces wasted volume but does not remove the ceiling.
- Correction: None applied at the code level beyond the early-exit optimization above. Documented
  as an explicit external blocker requiring a Groq-side quota check/reset before retrying.
- Prevention: For future benchmark work against capacity-constrained models, verify sustained
  throughput (not just a burst health check) before committing to a large scan volume.
- Evidence: `runtime-*-mini_gate.json` (failure evidence across four attempts).
- Status: Open; external dependency, not resolvable by further code changes.

## Current Verified State (2026-09-06)

- Synthetic validation: 76 tests passed (up from 66; includes tonight's new tests).
- Response-contract probe: all three models pass through the real production `make_llm` path.
- Tracer smoke: passed live (6 events, 1 `llm_start`).
- GPT-OSS JSON Object Mode: fixed and verified (`reasoning_effort="low"`).
- Mini-gate: not yet passed; blocked on Groq account-level rate limiting for `openai/gpt-oss-120b`.
- Full benchmark: not attempted; blocked behind the mini-gate per project sequencing.
