# LabelingOpsAgent Design — End-to-End Labeling and SFT Orchestration

> **Status**: Proposal. Not implemented.
> **Target**: Add a recoverable, auditable, policy-constrained Agent orchestration layer above the existing `labeling` and `labeling_sft` workflows.

## 1. Problem and Scope

The repository already provides two deterministic workflows:

- `labeling.bootstrap.BootstrapOrchestrator`: discovery, deduplication, labeling, validation, repair, and persistence.
- `labeling_sft.SFTOrchestrator`: dataset construction, training, and export.

What is missing is the cross-workflow control loop: use the number and quality of labels, outstanding reviews, training artifacts, and evaluation metrics to decide whether to collect more labels, train, review data, or release a candidate model.

`LabelingOpsAgent` is responsible for **planning and explaining the next step**. It must not directly perform labeling, training, or release operations. Deterministic code remains responsible for execution, data consistency, and hard policy gates.

## 2. Core Principles

1. **External state, not model memory**: the LLM has no persistent workflow memory. Every decision is based on durable `RunState`, reports, and policy.
2. **Agent decides; Controller executes**: the Agent can propose only allow-listed actions. `WorkflowController` validates preconditions, executes tools, and persists results.
3. **Hard gates cannot be bypassed**: for example, a model cannot be released without a passing evaluation report, and training cannot start from an unfrozen dataset.
4. **Side effects require approval**: paid API labeling, substantial training runs, and release or replacement of a production candidate require explicit human approval.
5. **Full provenance**: every exported model must be traceable to a label snapshot, prompt version, training configuration, evaluation report, and approval record.

## 3. Architecture

```text
User request / scheduled trigger
              |
              v
    +-----------------------+
    |    LabelingOpsAgent   |
    | plan, analyze, explain|
    +-----------+-----------+
                | allowed action proposal
                v
    +-----------------------+
    |  WorkflowController   |
    | policy / approval gate|
    +-----------+-----------+
                |
     +----------+----------+-----------+-----------+
     v                     v           v           v
 BootstrapOrchestrator  DatasetBuilder Trainer   Evaluator / Exporter
       (labeling)      (labeling_sft)  (SFT)       (labeling_sft)
                |
                v
        RunState + artifact / metric records
```

The first version should not introduce multiple autonomous sub-agents. A single supervisor Agent with controlled tools is easier to test, cheaper to operate, and better suited to an auditable training workflow.

## 4. Complete State Machine

```text
 +------------------+
 |   IDLE / INIT    |  create or resume RunState
 +---------+--------+
           |
           v
 +------------------+
 |  INSPECT_STATE   |  labels / datasets / models / metrics
 +----+--------+----+
      |        |                 \
      |        |                  \ existing candidate model
      |        |                   v
      |        |              +----------+
      |        |              | EVALUATE |
      |        |              +----+-----+
      |        |                   |
      |        v                   |
      |   +---------------+        |
      |   | BUILD_DATASET |        |
      |   +-------+-------+        |
      |           |                |
      |           v                |
      |   +---------------+        |
      |   | DATASET_CHECK |        |
      |   +-------+-------+        |
      |           |                |
      |           v                |
      |   +---------------+        |
      |   |     TRAIN     |--------+
      |   | checkpointing |        |
      |   +---------------+        |
      |                            v
      |                     +--------------+
      |                     | METRIC_GATE  |
      |                     +--+--------+--+
      |                        |        |
      |                     fail      pass
      |                        |        |
      |                        v        v
      |               +---------------+  +---------+
      |               | ERROR_ANALYSIS|  | RELEASE |
      |               +-------+-------+  +----+----+
      |                       |               |
      |                       +----+          v
      |                            |   +--------------------------+
      |                            +-->| EXPORT / DEPLOY_CANDIDATE|
      |                                +--------------------------+
      v
 +------------------+
 |  PLAN_LABELING   |
 +---------+--------+
           |
           v
 +------------------+
 | RESERVE_HOLDOUT  |  deduplicated holdout before training
 +---------+--------+
           |
           v
 +------------------+
 | BOOTSTRAP_LABEL  |  label, validate, repair, persist
 +---------+--------+
           |
           v
 +------------------+
 |   QUALITY_GATE   |
 +----+--------+----+
      |        |
   pass      review / fail
      |        |
      v        v
 +---------------+  +------------------+
 | ACCEPT_LABELS |  | REVIEW_REQUIRED  |
 +-------+-------+  +--------+---------+
         |                   |
         +--> BUILD_DATASET  +--> HUMAN_REVIEW --> ACCEPT_LABELS / REJECT_LABEL
```

`ERROR_ANALYSIS` can return the workflow to different states according to the error category:

```text
prompt coverage issue  -> update prompt version -> PLAN_LABELING
low-quality labels     -> targeted labeling + HUMAN_REVIEW
training issue         -> change approved training config -> BUILD_DATASET / TRAIN
```

## 5. State and Policy

`RunState` is the single source of truth for workflow runtime state. It should be stored in SQLite (recommended) or atomically written JSON. The model restores context from this state rather than relying on conversation history.

```json
{
  "run_id": "recipe-labeling-v1",
  "phase": "evaluate",
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
  "next_allowed_actions": ["review_errors", "plan_targeted_labeling"]
}
```

Initial policy gates should be configurable rather than hard-coded:

| Gate | Required condition | Failure action |
|---|---|---|
| `QUALITY_GATE` | Label validity rate, low-confidence ratio, and minimum accepted label count meet requirements | Review or targeted labeling |
| `DATASET_CHECK` | Label snapshot is frozen; train and validation splits are non-empty and reproducible | Repair the dataset |
| `METRIC_GATE` | Holdout schema validity, field F1, and other release metrics meet thresholds | Analyze errors and loop back |
| `RELEASE` | `METRIC_GATE` passes and a human approval record exists | Export and register a candidate model |

## 6. Proposed Package Structure

```text
src/labeling_ops/
├── agent.py            # LLM planner: reads state and proposes allowed actions
├── controller.py       # executes approved actions and enforces policy
├── contracts.py        # RunState, Task, Decision, MetricGate, ArtifactRecord
├── state_store.py      # SQLite-backed durable state and audit history
├── policies.py         # transitions, thresholds, approval requirements
├── run.py              # CLI/API composition root
└── tools/
    ├── labeling.py     # calls BootstrapOrchestrator
    ├── dataset.py      # calls a dataset builder
    ├── training.py     # calls SFTOrchestrator or a trainer
    ├── evaluation.py   # calls QwenEvaluator
    └── review.py       # creates and resolves review tasks
```

## 7. Agent Tool Contract

The Agent receives a compact state summary, the applicable policy, and a set of tool schemas. It returns a structured action proposal and never receives raw shell access.

```text
inspect_state(run_id)
plan_labeling(run_id, target, sampling_strategy)
run_bootstrap_labeling(run_id, target, model_backend)
create_review_tasks(run_id, filter)
build_dataset(run_id, label_snapshot)
train(run_id, config_version)
evaluate(run_id, model_artifact, holdout_version)
export_candidate(run_id, model_artifact, export_format)
```

The Controller rejects an action when it is absent from `next_allowed_actions`, violates a policy gate, lacks an approval record, or does not match the current `RunState` version.

## 8. Why Reinforcement Learning Is Not Needed Initially

The task sequence is a constrained workflow, not a skill that the LLM must memorize. Its reliable implementation is persistent state, deterministic transition rules, and controlled tool execution.

SFT remains responsible for teaching the labeling model to extract recipe JSON. Reinforcement learning can be considered later only after enough historical trajectories exist:

```text
state -> agent decision -> executed action -> evaluation / cost / review outcome
```

Those records could eventually optimize targeted sampling strategies or training configuration choices. They are not required to implement safe, resumable orchestration.

## 9. Phase-One Acceptance Criteria

1. A request such as “train a releasable recipe labeling model” creates a persisted plan.
2. Restarting at any phase resumes safely without duplicate labeling or duplicate training.
3. An evaluation failure cannot produce an export; it must create an auditable remediation plan.
4. Every exported GGUF traces to a label snapshot, prompt version, training configuration, and evaluation report.
5. Paid labeling, substantial training, and release require explicit approval.
