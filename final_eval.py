"""Fixed, resumable 60-task EB-Nav evaluation, without Teacher or Analyzer."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import reproduce as base
from seenav_rethink.assets import model_files
from seenav_rethink.data import check, load_json
from seenav_rethink.protocol import MARKER, decode_decision, marker_token_id


PROTOCOL = {
    "benchmark": "EmbodiedBench-Navigation base",
    "episodes": 60, "max_decisions": 20, "max_prompt_tokens": 3072, "max_new_tokens": 1024,
    "view_resolution": [500, 500], "dual_view": "BEV left, FV right", "fov": 100, "history_window": 5,
    # Allows the Qwen processor's 504x1008 aligned frame. Bounds vLLM's
    # synthetic warm-up input, which otherwise uses the 12.8M-pixel model max.
    "max_image_pixels": 1024 * 512,
    "move_m": 0.25, "rotate_deg": 90, "tilt_deg": 30, "success_distance_m": 1.0,
    "temperature": 0.4, "top_p": 1.0, "top_k": -1, "seed": 0, "hints": False,
    "inference": "vLLM V0/XFORMERS, TP=1",
    "tag_protocol": "strict_tokens_v2",
}
ENV_DIR = Path(base.LOCK["navigation_data"]).parent


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def task_list(source):
    path = base.verify_source(source) / base.LOCK["navigation_data"] / "base.json"
    tasks = load_json(path)["tasks"]
    ordered = [digest(task) for task in tasks]
    check(len(set(ordered)) == len(ordered) == PROTOCOL["episodes"], "Expected 60 distinct locked tasks.")
    return tasks, ordered


def eval_contract(source, ordered, output_format):
    return {
        **PROTOCOL, "output_format": output_format, "official_commit": base.LOCK["commit"],
        "tasks_sha256": base.LOCK["evaluation_task_sha256"], "ordered_task_sha256": ordered,
        "source_config_sha256": base.sha(source / ENV_DIR / "configs" / "config_ebnav.yaml"),
    }


def checkpoint_sha(model):
    weights = model_files(model)
    files = weights | {p.name for p in model.iterdir() if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"}}
    return {name: base.sha(model / name) for name in sorted(files)}


def atomic_json(path, value):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def episode_path(out, index):
    return out / f"episode_{index:03d}.json"


def validate_episode(row, index, task_hash, contract_sha):
    check((row["task_index"], row["task_sha256"], row["contract_sha256"]) == (index, task_hash, contract_sha)
          and type(row["success"]) is bool
          and 1 <= len(row["decisions"]) <= PROTOCOL["max_decisions"]
          and row["success"] == row["decisions"][-1]["won"], f"Invalid or mixed evaluation record: {index}")
    for step, decision in enumerate(row["decisions"], start=1):
        check(decision["step"] == step and decision["action"] in range(-1, 8)
              and decision["executed_action"] == decision["action"]
              and decision["protocol_valid"] == (decision["action"] != -1)
              and (not decision["valid_rethink"] or (decision["protocol_valid"] and decision["marker_count"] == 1)),
              f"Invalid action provenance for task {index}, step {step}")
    return row


def summarize(out, contract):
    signature = digest(contract)
    ordered = contract["ordered_task_sha256"]
    check(len(ordered) == len(set(ordered)) == PROTOCOL["episodes"], "Not the full fixed evaluation.")
    expected = {episode_path(out, i).name for i in range(len(ordered))}
    check({p.name for p in out.glob("episode_*.json")} == expected, "Expected exactly 60 evaluation records.")
    rows = [validate_episode(load_json(episode_path(out, i)), i, task_hash, signature)
            for i, task_hash in enumerate(ordered)]
    decisions = [d for row in rows for d in row["decisions"]]
    counts = {
        "protocol_valid": sum(d["protocol_valid"] for d in decisions),
        "marker_decisions": sum(d["marker_count"] > 0 for d in decisions),
        "valid_rethink_decisions": sum(d["valid_rethink"] for d in decisions),
        "action_successes": sum(d["action_success"] for d in decisions),
        "length_truncated": sum(d["finish_reason"] == "length" for d in decisions),
    }
    wins = sum(r["success"] for r in rows)
    return {
        "episodes": len(rows), "successes": wins, "success_rate": wins / len(rows), "decisions": len(decisions),
        **counts, "rates_per_decision": {k: v / len(decisions) for k, v in counts.items()},
        "contract_sha256": signature, "tasks_sha256": contract["tasks_sha256"],
        "episode_sha256": {name: base.sha(out / name) for name in sorted(expected)},
    }


def verify_result(out, source, output_format, *, model=None, require_marker=False):
    contract = load_json(out / "contract.json")
    _, ordered = task_list(source)
    expected = eval_contract(source, ordered, output_format)
    check(all(contract.get(k) == v for k, v in expected.items())
          and (not require_marker or type(contract.get("marker_token_id")) is int), "Wrong evaluation contract.")
    if model is not None:
        check(contract.get("model_sha256") == checkpoint_sha(model), "Evaluation used a different checkpoint.")
    result = summarize(out, contract)
    check(result == load_json(out / "summary.json"), "Evaluation summary does not match its episodes.")
    return result


class FixedIndex(int):
    def __bool__(self):
        # The pinned reset uses "if val_idx"; ordinary 0 selects a random task.
        return True


def check_reset(env, task):
    md = env.env.last_event.metadata
    agent, pose = md["agent"], task["agentPose"]
    pos = pose.get("position", pose)
    check(md["lastActionSuccess"] and md["sceneName"].removesuffix("_physics") == task["scene"].removesuffix("_physics")
          and env.target_object_id == task["targetObjectIds"] and env.target_position == task["target_position"]
          and env.user_instruction == task["instruction"]
          and all(math.isclose(agent["position"][k], pos[k], rel_tol=0., abs_tol=.02) for k in ("x", "z"))
          and abs((agent["rotation"]["y"] - pose["rotation"] + 180) % 360 - 180) < 1
          and abs(agent["cameraHorizon"] - pose["horizon"]) < 1, "Simulator did not reset to the locked task/pose.")


def eval_config(cfg, output_format):
    cfg["output_format"] = output_format
    c, d = cfg["envs"], cfg["dataset"]
    check(c["visual_prompt"] and c["max_steps"] == PROTOCOL["max_decisions"]
          and [c["width"], c["height"]] == PROTOCOL["view_resolution"]
          and c["fieldOfView"] == PROTOCOL["fov"] and c["history_window"] == PROTOCOL["history_window"]
          and c["moveMagnitude"] == PROTOCOL["move_m"] and c["rotateDegree"] == PROTOCOL["rotate_deg"]
          and d["eval_dataset"] == "base.json" and d["sueeess_threshold"] == PROTOCOL["success_distance_m"],
          "Pinned EB-Nav environment differs from the final evaluation contract.")
    # Standalone author evaluator / Appendix A.1 use 30°. The public *training*
    # YAML uses 90°; set this only in the independent evaluator, for both formats.
    c["tiltDegree"] = PROTOCOL["tilt_deg"]
    return cfg


def model_input(tokenizer, text, image, process_image):
    check(text.count("<image>") == 1 and MARKER not in text, "Unexpected native evaluation prompt.")
    chat = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
    chat = chat.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    frame = process_image(image)
    width, height = PROTOCOL["view_resolution"]
    check(frame.size == (width * 2, height), "Unexpected size for the native dual view.")
    return {"prompt_token_ids": tokenizer.encode(chat, add_special_tokens=False),
            "multi_modal_data": {"image": frame}}


def response_decision(tokenizer, token_ids, marker, output_format, json_projection):
    if output_format == "tags":
        decoded = decode_decision(tokenizer, token_ids, marker)
        return decoded.text, decoded.action, decoded.marker_count, decoded.routed
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    actions, valids = json_projection([text])
    action = actions[0]
    return (text, int(action) if valids[0] and type(action) in (int, float) and action in range(8) else -1,
            token_ids.count(marker) if marker is not None else 0, False)


def run_episodes(out, tasks, contract, pending, make_env, generate, make_input, interpret):
    """Independent environments and request seeds; no optimizer or hint client."""
    signature = digest(contract)
    batch = contract["batch_size"]
    with ExitStack() as stack:
        envs = []
        for _ in range(min(batch, len(pending))):
            env = make_env()
            stack.callback(env.close)
            envs.append(env)
        for offset in range(0, len(pending), batch):
            active = {}
            for env, index in zip(envs, pending[offset:offset+batch]):
                env.history_act_feedback.clear()
                text, image, _ = env.reset(val_idx=FixedIndex(index))
                check_reset(env, tasks[index])
                active[index] = (env, text, image, [])
            for step in range(PROTOCOL["max_decisions"]):
                if not active:
                    break
                indexes = list(active)
                inputs = [make_input(active[i][1], active[i][2]) for i in indexes]
                seeds = [int.from_bytes(hashlib.sha256(f"paper-seed0:{i}:{step}".encode()).digest()[:4], "big")
                         for i in indexes]
                for index, row in zip(indexes, generate(inputs, seeds), strict=True):
                    check(len(row.prompt_token_ids) <= PROTOCOL["max_prompt_tokens"], "Prompt exceeds native limit.")
                    response = row.outputs[0]
                    raw, action, count, routed = interpret(list(response.token_ids))
                    env, _, _, decisions = active[index]
                    text, image, _, done, info = env.step(action)
                    distance = float(info["distance"])
                    check(math.isfinite(distance) and int(info["action_id"]) == action,
                          "Simulator returned an invalid or mismatched action result.")
                    decisions.append({
                        "step": step + 1, "response": raw, "tokens": len(response.token_ids),
                        "protocol_valid": action != -1, "action": action, "executed_action": int(info["action_id"]),
                        "marker_count": count, "valid_rethink": routed, "finish_reason": response.finish_reason,
                        "action_success": bool(info["last_action_success"]), "feedback": str(info["env_feedback"]),
                        "distance_m": distance, "won": bool(info["won"]),
                    })
                    if done or step + 1 == PROTOCOL["max_decisions"]:
                        path = episode_path(out, index)
                        check(not path.exists(), "Refusing to overwrite a finished evaluation episode.")
                        atomic_json(path, {"task_index": index, "task_sha256": contract["ordered_task_sha256"][index],
                                          "contract_sha256": signature, "scene": tasks[index]["scene"],
                                          "success": bool(info["won"]), "decisions": decisions})
                        del active[index]
                        print(f"EB-Nav {index + 1:02d}/60 finished: success={int(info['won'])}", flush=True)
                    else:
                        active[index] = (env, text, image, decisions)


def pending_episodes(out, contract):
    pending, signature = [], digest(contract)
    for i, task_hash in enumerate(contract["ordered_task_sha256"]):
        path = episode_path(out, i)
        if path.exists():
            validate_episode(load_json(path), i, task_hash, signature)
        else:
            pending.append(i)
    return pending


def evaluate(source, out, model, output_format, batch_size, require_marker, tasks, ordered):
    from filelock import FileLock
    from transformers import AutoTokenizer

    check(sys.platform.startswith("linux"), "Final evaluation needs Linux CUDA and AI2-THOR CloudRendering.")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    check(not require_marker or (output_format == "tags" and marker is not None), "Expected the trained tag marker.")
    contract = {**eval_contract(source, ordered, output_format), "batch_size": batch_size, "dtype": "bfloat16",
                "marker_token_id": marker, "model_sha256": checkpoint_sha(model), "packages": base.package_versions(),
                "evaluator_sha256": base.text_sha(Path(__file__)),
                "tag_protocol_sha256": base.text_sha(base.ROOT / "seenav_rethink/protocol.py")}
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".eval.lock"), timeout=0):
        binding = out / "contract.json"
        if binding.exists():
            check(load_json(binding) == contract, "Cannot resume evaluation with a different contract/model.")
        else:
            check(not list(out.glob("episode_*.json")), "Missing contract for existing evaluation episodes.")
            atomic_json(binding, contract)
        pending = pending_episodes(out, contract)
        if pending:
            sys.path.insert(0, str(source / "verl-agent"))
            os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
            os.environ["VLLM_USE_V1"] = "0"
            from time import monotonic, sleep

            import torch
            from agent_system.environments.env_package.ebnav.envs import EBNavBaseEnv, load_config_file
            from agent_system.environments.env_package.ebnav.projection import ebnav_projection_json
            from agent_system.multi_turn_rollout.utils import process_image
            from vllm import LLM, SamplingParams

            check(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
                  "Expose exactly one allocated BF16 CUDA GPU for evaluation.")
            deadline = monotonic() + 90
            while True:
                free, total = torch.cuda.mem_get_info()
                if free >= .70 * total:
                    break
                check(monotonic() < deadline, "Evaluation GPU did not free sufficient memory; resume after it is idle.")
                sleep(2)
            cfg = eval_config(load_config_file(source / ENV_DIR / "configs" / "config_ebnav.yaml"), output_format)
            engine = None

            def generate(inputs, seeds):
                nonlocal engine
                if engine is None:  # Check the first real task reset before model allocation.
                    engine = LLM(model=str(model), dtype=contract["dtype"], tensor_parallel_size=1,
                                 max_model_len=PROTOCOL["max_prompt_tokens"] + PROTOCOL["max_new_tokens"],
                                 max_num_seqs=batch_size, gpu_memory_utilization=0.60,
                                 limit_mm_per_prompt={"image": 1, "video": 0},
                                 mm_processor_kwargs={"max_pixels": PROTOCOL["max_image_pixels"]},
                                 generation_config="vllm",
                                 trust_remote_code=False, seed=PROTOCOL["seed"])
                params = [SamplingParams(temperature=PROTOCOL["temperature"], top_p=PROTOCOL["top_p"],
                                         top_k=PROTOCOL["top_k"], max_tokens=PROTOCOL["max_new_tokens"],
                                         seed=seed, detokenize=False) for seed in seeds]
                return engine.generate(prompts=inputs, sampling_params=params, use_tqdm=False)

            run_episodes(out, tasks, contract, pending,
                         lambda: EBNavBaseEnv(cfg, is_train=False).init_env(), generate,
                         lambda text, image: model_input(tokenizer, text, image, process_image),
                         lambda ids: response_decision(tokenizer, ids, marker, output_format, ebnav_projection_json))
        result = summarize(out, contract)
        atomic_json(out / "summary.json", result)
        print(f"EB-Nav final: {result['successes']}/60 ({result['success_rate']:.2%}); {out / 'summary.json'}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--source", type=Path, default=base.DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, help="Local HF export, not the FSDP shard directory")
    parser.add_argument("--output", type=Path, help="Evaluation directory, reusable with the same contract")
    parser.add_argument("--format", choices=("tags", "json"), default="tags")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8), default=4)
    parser.add_argument("--require-marker", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    tasks, ordered = task_list(source)
    if args.command == "plan":
        print(json.dumps(eval_contract(source, ordered, args.format), indent=2, ensure_ascii=False))
        return
    if args.model is None or args.output is None:
        parser.error("run requires --model and --output")
    out, model = args.output.resolve(), args.model.resolve()
    check(not any(out.is_relative_to(p) for p in (source, model)), "Evaluation output must be outside source/model.")
    evaluate(source, out, model, args.format, args.batch_size, args.require_marker, tasks, ordered)


if __name__ == "__main__":
    main()
