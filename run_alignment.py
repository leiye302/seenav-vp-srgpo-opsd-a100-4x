"""Matched SeeNav baseline: original algorithm, strict tag actions, fixed final evaluation."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import reproduce as base
from final_eval import PROTOCOL, atomic_json, verify_result
from seenav_rethink.assets import model_files
from seenav_rethink.data import check, load_json

BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
STEPS = 150
RUNTIME = base.ROOT / ".deps" / "SeeNav-Agent-tags-v2"
PATCH = base.ROOT / "patches" / "tags-rollout.patch"
ROLLOUT = "verl-agent/agent_system/multi_turn_rollout/rollout_loop.py"
OVERLAY = {ROLLOUT: "694880fa7bacf87e4c30d51ce5cb3ad3e94e0ae8743ab109150ee3dee1a2fdd7"}
HAS_OPSD = "sdar" in base.LOCK
METHOD = base.LOCK.get("method", "VP + SRGPO + OPSD" if HAS_OPSD else "VP + SRGPO")


def verify_runtime(runtime):
    check((runtime / ".git").exists(), "Run python reproduce.py fetch to prepare the tagged runtime.")
    if HAS_OPSD:
        base.verify_opsd_source(runtime, extra_files=OVERLAY)
    else:
        check(base.git(runtime, "rev-parse", "HEAD") == base.LOCK["commit"], "Wrong SeeNav runtime commit.")
        changed = set(base.git(runtime, "diff", "--name-only", "HEAD").splitlines())
        check(changed == set(OVERLAY), "Unexpected files changed in tagged runtime.")
        for path, expected in OVERLAY.items():
            contents = (runtime / path).read_bytes().replace(b"\r\n", b"\n")
            check(hashlib.sha256(contents).hexdigest() == expected, f"Unexpected rollout contents: {path}")
    base.git(runtime, "apply", "--unidiff-zero", "--reverse", "--check", str(PATCH))
    return runtime


def fetch_runtime(source, target=RUNTIME):
    base.verify_source(source)
    check(target.resolve() != source.resolve(), "Keep the upstream checkout separate from the tagged runtime.")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tags-fetch-", dir=target.parent) as tmp:
            staging = Path(tmp) / "checkout"
            base.git(Path(tmp), "clone", "--no-hardlinks", "--no-checkout", str(source), str(staging))
            base.git(staging, "config", "core.autocrlf", "false")
            base.git(staging, "remote", "set-url", "origin", base.LOCK["repository"])
            base.git(staging, "checkout", "--detach", base.LOCK["commit"])
            if HAS_OPSD:
                base.git(staging, "apply", str(base.OPSD_PATCH))
            base.git(staging, "apply", "--unidiff-zero", str(PATCH))
            verify_runtime(staging)
            staging.rename(target)
    return verify_runtime(target)


def effective_launcher(native):
    script = base.isolated_paths(base.four_gpu_launcher(native))
    if HAS_OPSD:
        script = base.enable_opsd(script)
    script = base.replace_one(script, "actor_rollout_ref.actor.output_format=json",
                              "actor_rollout_ref.actor.output_format=tags")
    script = base.replace_one(script, f"actor_rollout_ref.model.path={base.LOCK['base_model']}",
                              'actor_rollout_ref.model.path="${SEENAV_BASE_MODEL}"')
    if "algorithm.adv_estimator=gigpo " in script:
        script = base.replace_one(script, "data.filter_overlong_prompts=False", "data.filter_overlong_prompts=True")
        line = "    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 " + chr(92) + "\n"
        script = base.replace_one(
            script, line, line + "    actor_rollout_ref.actor.over_long_response_penalty_coef=0.1 " + chr(92) + "\n",
        )
    check("data.filter_overlong_prompts=True" in script
          and "actor_rollout_ref.actor.over_long_response_penalty_coef=0.1 " in script,
          "Expected the matched SeeNav filter and length penalty.")
    return script


def pinned_model():
    from huggingface_hub import snapshot_download

    model = Path(snapshot_download(
        repo_id=base.LOCK["base_model"], revision=BASE_REVISION,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], token=False,
    )).absolute()
    check(model.name == BASE_REVISION, "Hub did not return the locked base revision.")
    check(not any(c.isspace() or c in ',=[]{}"\'\\' for c in model.as_posix()),
          "Put HF_HOME in a path without whitespace or Hydra metacharacters.")
    check(all((model / name).stat().st_size > 0 for name in model_files(model)), "Base weight files are empty.")
    return model


def protocol_binding():
    return {
        "method": METHOD,
        "actor_init": "pinned_base",
        "base_model_revision": BASE_REVISION,
        "output_format": "tags",
        "tag_protocol": PROTOCOL["tag_protocol"],
        "local_hints": False,
        "stage2_updates_expected": STEPS,
        "final_eval_tasks": PROTOCOL["episodes"],
        "rollout_overlay_sha256": OVERLAY[ROLLOUT],
    }


def evaluation_device(requested, visible):
    value = requested or visible or ""
    parts = [part.strip() for part in value.split(",")]
    check(requested is None or len(parts) == 1, "For final-eval, select one allocated GPU.")
    device = parts[0]
    check(re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", device), "Select an allocated evaluation GPU.")
    if visible is not None:
        check(device in {part.strip() for part in visible.split(",")}, "Evaluation GPU is not allocated.")
    return device


def training_result(out, source):
    check(load_json(out / "exit.json")["returncode"] == 0, "Training did not exit cleanly.")
    binding = load_json(out / "binding.json")
    expected = {**protocol_binding(), "official_commit": base.LOCK["commit"],
                "base_tasks_sha256": base.LOCK["evaluation_task_sha256"]}
    check(all(binding.get(key) == value for key, value in expected.items()), "Not a matched tag training run.")
    native = (source / base.LOCK["train_launcher"]).read_text(encoding="utf-8")
    check((out / "launch.sh").read_text(encoding="utf-8") == effective_launcher(native),
          "The recorded training launcher is not the aligned recipe.")
    root = out / "checkpoints"
    check((root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip() == str(STEPS),
          "Missing the expected step-150 checkpoint.")
    final = root / f"global_step_{STEPS}"
    actor = final / "actor"
    required = [final / "data.pt"] + [
        actor / f"{kind}_world_size_{base.GPU_COUNT}_rank_{rank}.pt"
        for rank in range(base.GPU_COUNT) for kind in ("model", "optim", "extra_state")
    ]
    check(all(p.is_file() and p.stat().st_size > 0 for p in required), "Step-150 training state is incomplete.")
    export = actor / "huggingface"
    check(all((export / name).stat().st_size > 0 for name in model_files(export)), "Final HF export is incomplete.")
    for step in range(1, STEPS + 1):
        metrics = load_json(out / "metrics" / f"metrics_{step}.json")
        check(metrics["training/global_step"] == step, "Training metrics do not match the expected updates.")
    return {
        "updates": STEPS,
        "output_format": "tags",
        "binding_sha256": base.sha(out / "binding.json"),
        "native_checkpoint": final.relative_to(out).as_posix(),
        "hf_export": export.relative_to(out).as_posix(),
    }


def finish(out, source, device):
    from filelock import FileLock

    with FileLock(str(out / ".finish.lock"), timeout=0):
        result = training_result(out, source)
        trained = out / "training_complete.json"
        if trained.exists():
            check(load_json(trained) == result, "Existing training receipt differs.")
        else:
            atomic_json(trained, result)
        model = out / result["hf_export"]
        evaluation = out / "final_eval"
        argv = [sys.executable, "-u", str(base.ROOT / "final_eval.py"), "run",
                "--source", str(source), "--model", str(model), "--output", str(evaluation), "--format", "tags"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("SEENAV_HINT_")}
        env["CUDA_VISIBLE_DEVICES"] = device
        logfile = out / "final_eval.log"
        print(f"Fixed EB-Nav 60-task final evaluation; follow {logfile}", flush=True)
        with logfile.open("ab") as log:
            finished = subprocess.run(argv, cwd=base.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if finished.returncode:
            raise SystemExit(f"Final evaluation exited with code {finished.returncode}; see {logfile}")
        summary = verify_result(evaluation, source, "tags", model=model)
        completed = out / "complete.json"
        payload = {**result, "final_eval": {
            "summary": "final_eval/summary.json", "episodes": summary["episodes"], "successes": summary["successes"],
            "success_rate": summary["success_rate"], "tasks_sha256": summary["tasks_sha256"],
            "contract_sha256": summary["contract_sha256"],
        }}
        if completed.exists():
            check(load_json(completed) == payload, "Existing completion receipt differs.")
        else:
            atomic_json(completed, payload)
        print(f"Verified step {STEPS} and final evaluation: {summary['successes']}/{PROTOCOL['episodes']}", flush=True)
