from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import final_eval as evaluation
import reproduce as base


SOURCE = Path(os.environ.get("SEENAV_TEST_SOURCE", base.DEFAULT_SOURCE))


class Env:
    def __init__(self, tasks, calls):
        self.history_act_feedback = []
        self.tasks, self.calls, self.closed = tasks, calls, False

    def reset(self, *, val_idx):
        if not val_idx:
            raise AssertionError("Author's if-val_idx branch would choose a random task.")
        self.index = int(val_idx)
        self.count = 0
        self.calls.append(("reset", self.index))
        task = self.tasks[self.index]
        pose = task["agentPose"]
        self.target_object_id, self.target_position = task["targetObjectIds"], task["target_position"]
        self.user_instruction = task["instruction"]
        self.env = SimpleNamespace(last_event=SimpleNamespace(metadata={
            "sceneName": task["scene"] + "_physics", "lastActionSuccess": True,
            "agent": {"position": pose.get("position", pose), "rotation": {"y": pose["rotation"]},
                      "cameraHorizon": pose["horizon"]},
        }))
        return str(self.index), self.index, {}

    def step(self, action):
        self.count += 1
        self.calls.append(("step", self.index, self.count, action))
        won = action == 2 and self.index % 2 == 0
        info = {"distance": .7 if won else 1.7, "action_id": action, "won": won,
                "last_action_success": action != -1, "env_feedback": ""}
        return str(self.index), self.index, won, won, info

    def close(self):
        self.closed = True


class EvaluationTest(unittest.TestCase):
    def test_pinned_60_ordered_tasks_shared_by_tag_and_json(self):
        tasks, ordered = evaluation.task_list(SOURCE)
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len(set(ordered)), 60)
        a = evaluation.eval_contract(SOURCE, ordered, "json")
        b = evaluation.eval_contract(SOURCE, ordered, "tags")
        self.assertEqual({**a, "output_format": "tags"}, b)
        self.assertEqual(a["tasks_sha256"], base.LOCK["evaluation_task_sha256"])
        self.assertFalse(a["hints"])

    def test_native_vp_prompt_without_forced_assistant_prefix(self):
        tok = Mock()
        tok.apply_chat_template.return_value = "<|im_start|>user\n\n<image>goal<|im_end|><|im_start|>assistant\n"
        tok.encode.return_value = [10, 11, 12]
        frame = SimpleNamespace(size=(1000, 500))
        self.assertEqual(evaluation.model_input(tok, "<image>goal", frame, lambda x: x),
                         {"prompt_token_ids": [10, 11, 12], "multi_modal_data": {"image": frame}})
        tok.apply_chat_template.assert_called_once_with([{"role": "user", "content": "<image>goal"}],
                                                       tokenize=False, add_generation_prompt=True)
        tok.encode.assert_called_once_with(
            "<|im_start|>user\n\n<|vision_start|><|image_pad|><|vision_end|>goal<|im_end|><|im_start|>assistant\n",
            add_special_tokens=False,
        )

    def test_real_json_projection_and_tag_validation_share_action_space(self):
        from test_alignment import Tokenizer

        path = SOURCE / evaluation.ENV_DIR / "projection.py"
        spec = importlib.util.spec_from_file_location("native_projection", path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        self.assertEqual(evaluation.response_decision(Tokenizer(), [12, 90, 13, 99], 90, "tags", None)[1:],
                         (2, 1, True))
        self.assertEqual(evaluation.response_decision(Tokenizer(), [12, 13, 90, 99], 90, "tags", None)[1:],
                         (-1, 1, False))
        tok = Mock()
        tok.decode.return_value = '{"executable_plan":[{"action_id":2},{"action_id":3}]}'
        self.assertEqual(evaluation.response_decision(tok, [4], None, "json", native.ebnav_projection_json)[1:],
                         (2, 0, False))

    def test_final_eval_geometry_is_separate_from_native_training(self):
        c = evaluation.PROTOCOL
        training = {
            "envs": {"visual_prompt": True, "width": 500, "height": 500, "max_steps": 20,
                     "fieldOfView": 100, "history_window": 5, "moveMagnitude": .25,
                     "rotateDegree": 90, "tiltDegree": 90},
            "dataset": {"eval_dataset": "base.json", "sueeess_threshold": 1},
        }
        for fmt in ("tags", "json"):
            cfg = evaluation.eval_config(deepcopy(training), fmt)
            self.assertEqual(cfg["envs"]["tiltDegree"], c["tilt_deg"])
            self.assertEqual(cfg["output_format"], fmt)
        self.assertEqual(training["envs"]["tiltDegree"], 90)

    def test_closed_loop_all_60_no_duplicate_resume_and_invalid_action_noop(self):
        from test_alignment import Tokenizer

        tasks, ordered = evaluation.task_list(SOURCE)
        contract = {**evaluation.eval_contract(SOURCE, ordered, "tags"), "batch_size": 4, "marker_token_id": 90}
        calls, instances = [], []

        def make_env():
            env = Env(tasks, calls)
            instances.append(env)
            return env

        def generate(indices, seeds):
            self.assertEqual(len(indices), len(seeds))
            return [
                SimpleNamespace(prompt_token_ids=[1, 2], outputs=[
                    SimpleNamespace(token_ids=([12, 90, 13, 99] if i % 2 == 0 else [14, 90, 13, 99]),
                                    finish_reason="stop")]) for i in indices
            ]

        def interpret(ids):
            return evaluation.response_decision(Tokenizer(), ids, 90, "tags", None)

        with TemporaryDirectory() as folder:
            out = Path(folder)
            evaluation.atomic_json(out / "contract.json", contract)
            evaluation.run_episodes(out, tasks, contract, evaluation.pending_episodes(out, contract)[:2],
                                    make_env, generate,
                                    lambda _text, idx: idx, interpret)
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                evaluation.summarize(out, contract)
            first_hash = base.sha(evaluation.episode_path(out, 0))
            self.assertEqual(evaluation.pending_episodes(out, contract), list(range(2, 60)))
            evaluation.run_episodes(out, tasks, contract, evaluation.pending_episodes(out, contract), make_env, generate,
                                    lambda _text, idx: idx, interpret)
            self.assertEqual(evaluation.pending_episodes(out, contract), [])
            self.assertEqual(first_hash, base.sha(evaluation.episode_path(out, 0)))
            self.assertEqual([c[1] for c in calls if c[0] == "reset"], list(range(60)))
            self.assertEqual([c[3] for c in calls if c[:2] == ("step", 1)], [-1]*20)
            self.assertTrue(all(env.closed for env in instances))
            result = evaluation.summarize(out, contract)
            self.assertEqual((result["episodes"], result["successes"], result["success_rate"]), (60, 30, .5))
            self.assertEqual(result["valid_rethink_decisions"], 30)
            self.assertEqual(result["marker_decisions"], 630)
            evaluation.atomic_json(out / "summary.json", result)
            self.assertEqual(evaluation.verify_result(out, SOURCE, "tags", require_marker=True), result)
            wrong = evaluation.load_json(evaluation.episode_path(out, 0))
            wrong["contract_sha256"] = "wrong"
            evaluation.atomic_json(evaluation.episode_path(out, 0), wrong)
            with self.assertRaisesRegex(ValueError, "mixed evaluation record"):
                evaluation.verify_result(out, SOURCE, "tags")
