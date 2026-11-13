import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import reproduce as base
import run_alignment as aligned
from seenav_rethink import protocol


class Tokenizer:
    marker = 90
    all_special_ids = [0, marker, 99]

    def encode(self, text, **kw):
        assert text == protocol.MARKER
        return [self.marker]

    def decode(self, ids, **kw):
        assert kw["skip_special_tokens"] is False
        return "".join({
            12: "<think>I should approach the marked chair.</think>",
            13: "<action>2</action>", 90: protocol.MARKER, 99: protocol.EOS,
            14: "<think></think>", 15: "<action>3</action>",
            16: "<RE", 17: "THINK>", 18: "\u00ad\n",
        }[i] for i in ids)


class MaskArray(np.ndarray):
    def bool(self):
        return self.astype(bool)


class AlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = base.verify_source(Path(os.environ.get("SEENAV_TEST_SOURCE", base.DEFAULT_SOURCE)))
        cls.native = (cls.source / base.LOCK["train_launcher"]).read_text(encoding="utf-8")

    def test_actual_rollout_gate_passes_only_valid_actions_without_marker_registration(self):
        runtime = aligned.verify_runtime(aligned.RUNTIME)
        tree = ast.parse((runtime / aligned.ROLLOUT).read_text(encoding="utf-8"))
        block = next(n.body[i:i+3] for n in ast.walk(tree) if isinstance(n, ast.For)
                     for i, stmt in enumerate(n.body)
                     if isinstance(stmt, ast.Assign)
                     and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                     and stmt.targets[0].id == "actor_config")
        tok = Tokenizer()
        tok.all_special_ids = [0, 99]  # the baseline does not register RETHINK
        # Good, empty, wrong order, repeated action, marker spelled by ordinary tokens.
        rows = [[12,18,13,99,0], [14,13,99,0,0], [13,12,99,0,0], [12,13,15,99,0], [12,16,17,13,99]]
        ids = np.array(rows).view(MaskArray)
        batch = SimpleNamespace(batch={
            "responses": ids,
            "attention_mask": np.pad(ids != 0, ((0, 0), (3, 0)), constant_values=True).view(MaskArray),
        })
        self_arg = SimpleNamespace(tokenizer=tok, config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(output_format="tags")),
        ))
        state = {"self": self_arg, "batch": batch}
        exec(compile(ast.Module(body=block, type_ignores=[]), aligned.ROLLOUT, "exec"), state)
        self.assertEqual(state["text_actions"][1:], [""] * 4)
        self.assertNotIn("\u00ad", state["text_actions"][0])
        spec = importlib.util.spec_from_file_location("native_projection", self.source / Path(base.LOCK["navigation_data"]).parent / "projection.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        moves, valid = native.ebnav_projection(state["text_actions"])
        self.assertEqual(valid, [1, 0, 0, 0, 0])
        self.assertEqual(moves[0], 2)
        np.testing.assert_array_equal(batch.batch["responses"], rows)

    def test_base_revision_is_passed_to_hub_and_checked_before_training(self):
        with TemporaryDirectory() as folder:
            model = Path(folder) / aligned.BASE_REVISION
            model.mkdir()
            for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
                (model / name).write_bytes(b"fixture")
            (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
            download = Mock(return_value=str(model))
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                if any(c.isspace() for c in str(model)):
                    with self.assertRaisesRegex(ValueError, "HF_HOME"):
                        aligned.pinned_model()
                else:
                    self.assertEqual(aligned.pinned_model(), model)
            download.assert_called_once_with(
                repo_id=base.LOCK["base_model"], revision=aligned.BASE_REVISION,
                allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], token=False,
            )
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=lambda **kw: str(model.parent))}):
                with self.assertRaisesRegex(ValueError, "locked base revision"):
                    aligned.pinned_model()

    def test_finished_training_only_runs_fixed_tag_evaluation_and_can_resume(self):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")

        with TemporaryDirectory() as folder:
            out = Path(folder)
            binding = {**aligned.protocol_binding(), "official_commit": base.LOCK["commit"],
                       "base_tasks_sha256": base.LOCK["evaluation_task_sha256"]}
            write(out / "exit.json", {"returncode": 0})
            write(out / "binding.json", {**binding, "output_format": "json"})
            with self.assertRaisesRegex(ValueError, "Not a matched tag training run"):
                aligned.training_result(out, self.source)
            write(out / "binding.json", binding)
            (out / "launch.sh").write_text(aligned.effective_launcher(self.native), encoding="utf-8")
            root = out / "checkpoints"
            actor = root / "global_step_150" / "actor"
            export = actor / "huggingface"
            write(export / "config.json", {"model_type": "qwen2_5_vl"})
            for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
                (export / name).write_bytes(b"fixture")
            for rank in range(base.GPU_COUNT):
                for kind in ("model", "optim", "extra_state"):
                    (actor / f"{kind}_world_size_{base.GPU_COUNT}_rank_{rank}.pt").write_bytes(b"fixture")
            missing = actor.parent / "data.pt"
            (root / "latest_checkpointed_iteration.txt").write_text("150", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training state is incomplete"):
                aligned.training_result(out, self.source)
            missing.write_bytes(b"fixture")
            for step in range(1, 151):
                write(out / "metrics" / f"metrics_{step}.json", {"training/global_step": step})
            summary = {"episodes": 60, "successes": 10, "success_rate": 1/6,
                       "tasks_sha256": base.LOCK["evaluation_task_sha256"], "contract_sha256": "fixture"}
            with patch.object(aligned.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                    patch.dict(os.environ, {"SEENAV_HINT_API_KEY": "not-used-by-evaluation"}):
                with self.assertRaisesRegex(SystemExit, "Final evaluation"):
                    aligned.finish(out, self.source, "2")
            self.assertTrue((out / "training_complete.json").is_file())
            self.assertFalse((out / "complete.json").exists())
            with patch.object(aligned.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child, \
                    patch.object(aligned, "verify_result", return_value=summary) as verify, \
                    patch.dict(os.environ, {"SEENAV_HINT_API_KEY": "not-used-by-evaluation"}):
                aligned.finish(out, self.source, "2")
            argv = child.call_args.args[0]
            self.assertEqual(Path(argv[2]).name, "final_eval.py")
            self.assertEqual(argv[-2:], ["--format", "tags"])
            self.assertNotIn("--require-marker", argv)
            self.assertEqual(child.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertNotIn("SEENAV_HINT_API_KEY", child.call_args.kwargs["env"])
            verify.assert_called_once_with(out / "final_eval", self.source, "tags", model=export)
            self.assertEqual(aligned.load_json(out / "complete.json")["final_eval"]["successes"], 10)

    def test_final_eval_cannot_select_unallocated_or_multiple_gpus(self):
        self.assertEqual(aligned.evaluation_device(None, "2,3,6,7"), "2")
        self.assertEqual(aligned.evaluation_device("7", "2,3,6,7"), "7")
        for requested in ("0", "2,3", "not-a-device"):
            with self.assertRaises(ValueError):
                aligned.evaluation_device(requested, "2,3,6,7")
