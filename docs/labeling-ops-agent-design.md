# LabelingOpsAgent Design - A StateFlow-Inspired Orchestration Workflow

> **Status**: Proposal. Not implemented.
> **Primary reference**: Wu et al., *StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows*, COLM 2024, [arXiv:2403.11322v5](https://arxiv.org/abs/2403.11322).
> **Target**: Add a recoverable, auditable, policy-constrained orchestration layer above the existing `labeling` and `labeling_sft` workflows.

## 1. Problem

The repository already contains deterministic components for individual phases:

- `labeling.bootstrap.BootstrapOrchestrator` discovers, deduplicates, labels, validates, repairs, and persists records.
- `labeling_sft.SFTOrchestrator` builds a dataset, trains a model, and exports an artifact.
- `OllamaEvaluator` evaluates a deployed Ollama model.

The missing piece is a cross-phase control loop. It must use label quality, review backlog, dataset versions, training artifacts, and evaluation outcomes to decide whether to collect more labels, perform review, train, evaluate, or release a candidate model.

This is a poor fit for one long prompt that asks an LLM to remember every workflow step. The model can lose track of progress, repeat actions, and make transitions that are difficult to audit. Instead, the system should model the workflow explicitly as a state machine, following the StateFlow separation between **process grounding** and **sub-task solving**.

## 2. StateFlow Mapping

StateFlow represents an LLM workflow as:

```text
<S, s0, F, delta, Gamma, Omega>
```

For LabelingOpsAgent:

| StateFlow element | LabelingOpsAgent meaning |
|---|---|
| `S` | Finite set of workflow states, such as `LABEL`, `REVIEW`, `TRAIN`, and `EVALUATE` |
| `s0` | `INIT` or a resumed persisted state |
| `F` | `DEPLOY_CANDIDATE`, `CANCELLED`, or `BLOCKED` |
| `delta` | Transition function based on durable state, tool observations, policy gates, and, only when necessary, an LLM decision |
| `Gamma` | Append-only execution history: user request, state prompts, Agent outputs, tool results, approvals, metrics, and errors |
| `Omega` | Output functions: prompt builders, LLM calls, deterministic checks, and calls to existing labeling/SFT tools |

The key distinction is:

```text
Process grounding:  state + transition rules + durable RunState
Sub-task solving:   state-specific prompt + LLM reasoning + tool execution
```

The LLM therefore does not own the workflow. It solves a focused sub-task after the Controller has entered a state; the Controller owns progress, permissions, and persistence.

## 3. Architecture

```text
User request / scheduled trigger
              |
              v
    +-----------------------+
    |  StateFlow Controller |
    | current state + delta |
    +-----------+-----------+
                |
                | enter state and run its output functions (Omega)
                v
    +-----------------------+
    |    LabelingOpsAgent   |
    | state-specific prompt |
    | plan / analyze / route|
    +-----------+-----------+
                |
                v
    +-----------------------+
    |  Controlled tool set  |
    +----+--------+----+----+
         |        |    |
         v        v    v
   labeling     SFT    evaluator / exporter
         |        |    |
         +--------+----+
                  |
                  v
      RunState + append-only event history (Gamma)
```

`LabelingOpsAgent` may propose an action only from the currently allowed actions. `WorkflowController` validates preconditions, approvals, and state version before executing it.

The first version should use one supervisor Agent, not multiple autonomous sub-agents. State-specific prompts provide the specialization described by StateFlow without adding coordination overhead.

## 4. State Machine

```text
                         +---------------+
                         | INIT / RESUME |
                         +-------+-------+
                                 |
                                 v
                         +---------------+
                         | INSPECT_STATE |
                         +---------------+

INSPECT_STATE guarded routes:

  no usable accepted labels ------------------------------> PLAN_LABELING
  usable labels but no valid dataset/candidate -----------> BUILD_DATASET
  candidate model has no current evaluation --------------> EVALUATE
  candidate has been exported and registered -------------> DEPLOY_CANDIDATE [final]

Labeling path:

  PLAN_LABELING -- holdout absent --> RESERVE_HOLDOUT -- manifest saved --> BOOTSTRAP_LABEL
        |
        +------ holdout exists --------------------------------------------> BOOTSTRAP_LABEL

  BOOTSTRAP_LABEL --> QUALITY_GATE -- pass --------> ACCEPT_LABELS --> INSPECT_STATE
                                      |
                                      +-- review --> REVIEW_REQUIRED --> HUMAN_REVIEW
                                                                     |
                                                                     +--> INSPECT_STATE

Dataset, training, and release path:

  BUILD_DATASET --> DATASET_CHECK -- pass --> TRAIN -- candidate artifact --> EVALUATE --> METRIC_GATE
                         |                      |                                                |
                         +-- fail --> BLOCKED    +-- training failure --> ERROR_ANALYSIS          +-- pass --> RELEASE
                              [final]                                                                  |
                                                                                                       v
                                                                                         EXPORT / DEPLOY_CANDIDATE [final]

  METRIC_GATE -- fail --> ERROR_ANALYSIS

ERROR_ANALYSIS guarded routes:

  prompt coverage issue ----------------------------------------> PLAN_LABELING
  label quality issue ------------------------------------------> REVIEW_REQUIRED or PLAN_LABELING
  training issue, approved configuration change ----------------> TRAIN
  no safe remediation ------------------------------------------> BLOCKED [final]

Any non-final state -- explicit user cancellation -------------> CANCELLED [final]
```

`INSPECT_STATE` routes only from persisted facts: label availability, dataset and candidate availability, evaluation freshness, and export status. A user cancellation transitions to the final `CANCELLED` state from any non-final state.

The state machine must include terminal failure states such as `BLOCKED` and `CANCELLED`, rather than retrying indefinitely.

### 4.1 Transition Table

The diagram is illustrative; this table is authoritative. Each row is a single legal transition.

| From state | Guard / event | To state |
|---|---|---|
| `INIT / RESUME` | State loaded or created | `INSPECT_STATE` |
| `INSPECT_STATE` | No usable accepted labels | `PLAN_LABELING` |
| `INSPECT_STATE` | Usable labels, no frozen dataset or candidate model | `BUILD_DATASET` |
| `INSPECT_STATE` | Candidate model exists and has no current evaluation | `EVALUATE` |
| `INSPECT_STATE` | Candidate passed release and was exported | `DEPLOY_CANDIDATE` |
| `PLAN_LABELING` | Holdout has not been reserved | `RESERVE_HOLDOUT` |
| `PLAN_LABELING` | Holdout already exists | `BOOTSTRAP_LABEL` |
| `RESERVE_HOLDOUT` | Holdout manifest persisted | `BOOTSTRAP_LABEL` |
| `BOOTSTRAP_LABEL` | Batch finishes | `QUALITY_GATE` |
| `QUALITY_GATE` | Quality policy passes | `ACCEPT_LABELS` |
| `QUALITY_GATE` | Review policy requires intervention | `REVIEW_REQUIRED` |
| `ACCEPT_LABELS` | Labels and provenance are persisted | `INSPECT_STATE` |
| `REVIEW_REQUIRED` | Review tasks created | `HUMAN_REVIEW` |
| `HUMAN_REVIEW` | Accepted or corrected review records persisted | `INSPECT_STATE` |
| `HUMAN_REVIEW` | Review records are rejected but the remaining accepted dataset is still viable | `INSPECT_STATE` |
| `HUMAN_REVIEW` | Review cannot be completed and policy requires intervention | `BLOCKED` |
| `BUILD_DATASET` | Dataset split created | `DATASET_CHECK` |
| `DATASET_CHECK` | Deterministic checks pass | `TRAIN` |
| `DATASET_CHECK` | Deterministic checks fail | `BLOCKED` |
| `TRAIN` | Candidate artifact produced | `EVALUATE` |
| `TRAIN` | Training cannot resume or fails | `ERROR_ANALYSIS` |
| `EVALUATE` | Evaluation report persisted | `METRIC_GATE` |
| `METRIC_GATE` | All release metrics pass | `RELEASE` |
| `METRIC_GATE` | At least one release metric fails | `ERROR_ANALYSIS` |
| `ERROR_ANALYSIS` | Prompt coverage issue | `PLAN_LABELING` |
| `ERROR_ANALYSIS` | Label quality issue | `REVIEW_REQUIRED` or `PLAN_LABELING` |
| `ERROR_ANALYSIS` | Approved training configuration change | `TRAIN` |
| `ERROR_ANALYSIS` | No safe remediation exists | `BLOCKED` |
| `RELEASE` | Human approval exists | `EXPORT / DEPLOY_CANDIDATE` |
| Any non-final state | Explicit user cancellation | `CANCELLED` |

## 5. State Definitions and Output Functions

Each state has a small, explicit sequence of output functions. In StateFlow terms, an output function may be a prompt function, LLM call, deterministic validator, or external tool call. Tool output is appended to the event history before `delta` selects the next state.

| State | State-specific objective | Output functions | Primary transition evidence |
|---|---|---|---|
| `INSPECT_STATE` | Establish the current run position | load `RunState`; load latest artifacts and metrics; summarize for Agent | persisted phase and artifact availability |
| `PLAN_LABELING` | Select a bounded labeling batch | Agent proposes target and sampling strategy; approval check | target approved; budget available |
| `RESERVE_HOLDOUT` | Isolate unseen evaluation data | deduplicate and reserve holdout | holdout manifest is durable |
| `BOOTSTRAP_LABEL` | Produce validated label records | call `BootstrapOrchestrator`; persist checkpoint and records | batch result, errors, and accepted count |
| `QUALITY_GATE` | Decide if labels are usable | deterministic quality metrics; optional Agent diagnosis | validity, confidence, coverage, review backlog |
| `REVIEW_REQUIRED` | Create review work | select low-confidence or inconsistent records | review tasks durable |
| `HUMAN_REVIEW` | Resolve quality-sensitive records | await human decision; persist correction or rejection | review decision |
| `BUILD_DATASET` | Freeze a reproducible training input | snapshot accepted labels; build train/validation split | snapshot ID and split statistics |
| `DATASET_CHECK` | Block invalid training inputs | deterministic schema and split checks | pass/fail report |
| `TRAIN` | Produce a checkpointed candidate | call `SFTOrchestrator` or trainer | training result and artifact locations |
| `EVALUATE` | Measure generalization | call `OllamaEvaluator` on holdout and regression sets | evaluation report |
| `METRIC_GATE` | Apply release policy | deterministic threshold checks | named metric values and policy version |
| `ERROR_ANALYSIS` | Propose a bounded remediation path | Agent classifies evidence; Controller validates proposal | categorized failure report |
| `RELEASE` | Authorize candidate creation | verify gate; require human approval | approval record |
| `EXPORT / DEPLOY_CANDIDATE` | Create a traceable deployable artifact | export GGUF; register candidate metadata | artifact checksum and provenance |

## 6. Transition Design

StateFlow describes two useful transition styles. Both apply here.

### 6.1 Deterministic transitions are the default

Use structured tool results and policy checks whenever possible. Examples:

```text
BOOTSTRAP_LABEL -> QUALITY_GATE
    when BootstrapOrchestrator returns a completed batch.

QUALITY_GATE -> REVIEW_REQUIRED
    when low-confidence ratio exceeds its configured threshold.

METRIC_GATE -> RELEASE
    only when every required metric passes and an approval exists.

METRIC_GATE -> ERROR_ANALYSIS
    when any release metric fails.
```

These transitions are auditable and cannot be overridden by a persuasive model response.

### 6.2 LLM-routed transitions are an exception

Use an LLM only when evidence is semantically ambiguous, primarily in `ERROR_ANALYSIS`. The Agent may classify a failure as prompt coverage, label quality, training configuration, or an unknown category. It must return structured JSON with evidence references and one proposed allowed action. The Controller then validates the action against policy.

```json
{
  "category": "label_quality",
  "evidence": ["eval-014: ingredient_refs mismatch", "review-backlog: 12"],
  "proposed_action": "create_review_tasks",
  "reason": "Validation failures cluster around labels that were not manually reviewed."
}
```

## 7. Persistent State and History

StateFlow treats the state plus cumulative context history as a snapshot of a running workflow. For this system, do not reconstruct that snapshot from chat history alone. Persist two complementary records.

### 7.1 `RunState`: compact operational truth

Store this in SQLite (recommended) and update it transactionally after every successful transition.

```json
{
  "run_id": "recipe-labeling-v1",
  "state": "evaluate",
  "state_version": 42,
  "dataset_version": "dataset-20260807-03",
  "prompt_version": "recipe-prompt-04",
  "accepted_labels": 516,
  "review_pending": 12,
  "training_run": "run-014",
  "artifacts": [".../adapter", ".../checkpoint-200"],
  "metrics": {
    "holdout_schema_validity": 0.94,
    "field_f1": 0.79
  },
  "next_allowed_actions": ["review_errors", "plan_targeted_labeling"],
  "transition_count": 9,
  "max_transition_count": 20
}
```

### 7.2 `EventHistory`: append-only evidence

Store one immutable record per prompt, tool invocation, output, approval, error, and transition. Retain references to large documents instead of injecting their full content into every LLM prompt. State-specific prompt builders should assemble only the relevant evidence, keeping context focused and controlling token cost.

`max_transition_count` is a StateFlow-style loop guard. Exceeding it transitions the run to `BLOCKED` and creates a human intervention task.

## 8. Policies and Non-Bypassable Gates

All thresholds must be versioned configuration, not prose embedded in prompts.

| Gate | Required condition | Failure action |
|---|---|---|
| `QUALITY_GATE` | Label validity rate, low-confidence ratio, coverage, and accepted-label count meet requirements | Review or targeted labeling |
| `DATASET_CHECK` | Label snapshot is frozen; train and validation sets are non-empty and reproducible | Repair the dataset |
| `METRIC_GATE` | Holdout schema validity, field F1, and other release metrics meet thresholds | Error analysis and controlled loopback |
| `RELEASE` | `METRIC_GATE` passes and an approval record exists | Export and register a candidate |

Paid API calls, major training runs, and model release require a pending approval state. This separates an Agent's recommendation from an authorized side effect.

## 9. Package Structure

```text
src/labeling_ops/
|-- agent.py            # state-specific LLM calls and structured recommendations
|-- controller.py       # StateFlow loop, output execution, and transition enforcement
|-- contracts.py        # RunState, Event, State, Decision, MetricGate, ArtifactRecord
|-- state_store.py      # SQLite state and append-only event history
|-- policies.py         # versioned thresholds, transition rules, approvals, loop limits
|-- prompts/
|   |-- inspect_state.txt
|   `-- error_analysis.txt
|-- tools/
|   |-- labeling.py     # BootstrapOrchestrator adapter
|   |-- dataset.py      # dataset builder adapter
|   |-- training.py     # SFTOrchestrator or trainer adapter
|   |-- evaluation.py   # OllamaEvaluator adapter
|   `-- review.py       # review task adapter
`-- run.py              # CLI/API composition root
```

The `controller.py` loop follows the StateFlow algorithm at an implementation level:

```text
state = load_or_create_run_state()
while state not in FINAL_STATES and state.transition_count < policy.max_transitions:
    events = execute_output_functions(state)
    append_events(events)
    state = transition(state, events, policy)
    persist(state)
```

## 10. Relationship to Training and Reinforcement Learning

The labeling SFT model and the orchestration Agent solve different problems:

| Component | Learns or controls |
|---|---|
| Labeling SFT model | Recipe Markdown -> structured JSON extraction |
| StateFlow Controller | Valid phase progression, recovery, policy gates, and tool execution |
| LabelingOpsAgent | Focused planning and error diagnosis within the current state |

Reinforcement learning is not required for the initial orchestration implementation. The primary reliability mechanism is the state machine plus persistent evidence and controlled tools.

StateFlow is compatible with iterative refinement. Once the system has accumulated auditable trajectories,

```text
state -> action proposal -> tool result -> metric / cost / review outcome
```

the team can analyze recurring failure states, refine state prompts, add or split states, and later evaluate learning-based routing or targeted sampling. Any such optimization must remain downstream of the deterministic policy gates.

## 11. Phase-One Acceptance Criteria

1. A request such as "train a releasable recipe labeling model" creates a durable `RunState` and event history.
2. Every state runs only its declared output functions and can resume safely after interruption.
3. The Controller rejects an action that is not valid for the current state or lacks required approval.
4. The workflow stops in `BLOCKED` after the configured transition limit instead of looping indefinitely.
5. Evaluation failure cannot produce an export; it creates an evidence-backed remediation path.
6. Every exported GGUF traces to a label snapshot, prompt version, training configuration, evaluation report, policy version, and approval record.
