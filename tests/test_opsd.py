"""CPU checks of the real native hook, position rule and sampled-token loss."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
import run_alignment
from seenav_opsd.teacher import attach_teacher_scores, pack_teacher_tensors


class Proto:
    def __init__(self, tensors, non_tensors=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors or {}

    @classmethod
    def from_dict(cls, tensors, non_tensors):
        return cls(tensors, non_tensors)


class OpsdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = reproduce.verify_source(Path(os.environ.get("SEENAV_TEST_SOURCE", reproduce.DEFAULT_SOURCE)))
        cls.runtime = run_alignment.verify_runtime(run_alignment.RUNTIME)
        cls.native = cls.source / "verl-agent"

    def load_function(self, relative, name, **symbols):
        path = self.native / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {"torch": torch, **symbols}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]

    def fixture(self):
        # One merged image per row (4 visual tokens), unequal left padding,
        # and unequal right padding in the response.
        ids = torch.tensor([
            [0, 0, 11, 12, 12, 12, 12, 13, 21, 22, 71, 72, 0],
            [0, 23, 11, 12, 12, 12, 12, 13, 24, 25, 73, 74, 75],
        ])
        mask = ids.ne(0).long()
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(merge_size=2),
            tokenizer=SimpleNamespace(convert_tokens_to_ids={
                "<|vision_start|>": 11, "<|image_pad|>": 12, "<|video_pad|>": 14,
            }.__getitem__),
        )
        grid = torch.tensor([[1, 4, 4]])
        rope = self.load_function("verl/models/transformers/qwen2_vl.py", "get_rope_index", Optional=Optional)
        positions = torch.stack([rope(processor, row, grid, attention_mask=m) for row, m in zip(ids, mask)])
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                "responses": ids[:, -3:].clone()}, rope, processor, grid

    def test_preserves_suffix_and_matches_native_rope_in_both_layouts(self):
        native, rope, processor, grid = self.fixture()
        for scalar in (False, True):
            student = dict(native)
            if scalar:
                student["position_ids"] = student["position_ids"][:, 0, :]
            copies = {key: value.clone() for key, value in student.items()}
            teacher = pack_teacher_tensors(student, [31, 32, 33], 64)
            self.assertTrue(torch.equal(teacher["responses"], student["responses"]))
            self.assertTrue(torch.equal(teacher["input_ids"][:, -3:], student["responses"]))
            self.assertTrue(torch.equal(teacher["attention_mask"][:, -3:], student["attention_mask"][:, -3:]))
            for i, left in enumerate((2, 1)):
                self.assertEqual(teacher["input_ids"][i, left:left + 3].tolist(), [31, 32, 33])
                self.assertTrue(torch.equal(teacher["input_ids"][i, left + 3:], student["input_ids"][i, left:]))
                expected = rope(processor, teacher["input_ids"][i], grid, attention_mask=teacher["attention_mask"][i])
                if scalar:
                    expected = expected[0]
                self.assertTrue(torch.equal(teacher["position_ids"][i], expected))
            for key, value in copies.items():
                self.assertTrue(torch.equal(student[key], value))

    def test_teacher_scoring_uses_same_worker_images_tokens_and_no_grad_output(self):
        native, _, _, grid = self.fixture()
        native["old_log_probs"] = torch.zeros_like(native["responses"], dtype=torch.float32)
        images = [{"pixel_values": torch.rand(16, 8), "image_grid_thw": grid} for _ in range(2)]
        batch = Proto(native, {"multi_modal_inputs": images, "raw_step_rewards": [1, 0]})
        outputs = torch.randn(2, 3, requires_grad=True)
        calls = []

        def score(value):
            calls.append(value)
            self.assertIs(value.non_tensor_batch["multi_modal_inputs"], images)
            self.assertEqual(set(value.non_tensor_batch), {"multi_modal_inputs"})
            self.assertTrue(torch.equal(value.batch["responses"], native["responses"]))
            self.assertTrue(torch.equal(value.batch["attention_mask"][:, -3:], native["attention_mask"][:, -3:]))
            self.assertEqual(value.batch["input_ids"].shape[-1], native["input_ids"].shape[-1] + 3)
            return Proto({"old_log_probs": outputs})

        def encode(text, add_special_tokens):
            self.assertFalse(add_special_tokens)
            self.assertTrue(text.startswith("[Privileged Skill Information]\n"))
            self.assertIn("return a single available action inside the native <action> tag.", text)
            self.assertNotIn("JSON", text)
            return [31, 32, 33]

        trainer = SimpleNamespace(
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(multi_turn=SimpleNamespace(enable=False), max_model_len=64))),
            tokenizer=SimpleNamespace(encode=encode, all_special_ids=[0, 11, 12, 13, 14]),
            actor_rollout_wg=SimpleNamespace(compute_log_prob=score),
        )
        verl = ModuleType("verl")
        verl.DataProto = Proto
        with patch.dict(sys.modules, {"verl": verl}):
            stats = attach_teacher_scores(trainer, batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats["sdar/teacher_rows"], 2)
        self.assertEqual(stats["sdar/teacher_prefix_tokens"], 3)
        self.assertTrue(torch.equal(batch.batch["teacher_log_probs"], outputs))
        self.assertFalse(batch.batch["teacher_log_probs"].requires_grad)

    def test_no_truncation_or_response_repair(self):
        student, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "context limit"):
            pack_teacher_tensors(student, [31, 32, 33], 13)
        student["responses"][0, 0] = 88
        with self.assertRaisesRegex(ValueError, "unchanged input suffix"):
            pack_teacher_tensors(student, [31, 32, 33], 64)

    def test_locked_loss_adds_only_detached_gated_gradient(self):
        masked_mean = self.load_function("verl/utils/torch_functional.py", "masked_mean")
        agg_loss = self.load_function("verl/trainer/ppo/core_algos.py", "agg_loss",
                                      verl_F=SimpleNamespace(masked_mean=masked_mean))
        core = ModuleType("verl.trainer.ppo.core_algos")
        core.agg_loss = agg_loss
        with patch.dict(sys.modules, {"verl.trainer.ppo.core_algos": core}):
            from seenav_opsd.loss import add_sdar_loss
            student = torch.tensor([[-1.5, -0.8, float("nan")]], dtype=torch.float64, requires_grad=True)
            teacher = torch.tensor([[-0.5, -1.3, float("nan")]], dtype=torch.float64, requires_grad=True)
            mask = torch.tensor([[1, 1, 0]])
            base = 0.3 * student[0, 0]
            cfg = SimpleNamespace(sdar_loss_coef=.01, sdar_gate_beta=5., loss_agg_mode="token-mean")
            total, stats = add_sdar_loss(base, student, teacher, mask, cfg)
            total.backward()
            expected = -.01 * torch.sigmoid(5 * (teacher.detach()[0, :2] - student.detach()[0, :2])) / (2 + 1e-8)
            expected[0] += .3
            torch.testing.assert_close(student.grad[0, :2], expected)
            self.assertEqual(student.grad[0, 2], 0)
            self.assertIsNone(teacher.grad)
            self.assertEqual(stats["sdar/gate_active_ratio"], .5)

    def test_native_validation_reward_and_scoring_boundaries(self):
        relative = "verl/trainer/ppo/ray_trainer.py"
        old = ast.parse((self.native / relative).read_text(encoding="utf-8"))
        text = (self.runtime / "verl-agent" / relative).read_text(encoding="utf-8")
        new = ast.parse(text)

        def find(tree, name):
            return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

        for name in ("_validate", "compute_advantage", "apply_invalid_action_penalty"):
            self.assertEqual(ast.dump(find(old, name)), ast.dump(find(new, name)))
        fit = ast.get_source_segment(text, find(new, "fit"))
        self.assertLess(fit.index("batch = compute_advantage("), fit.index("attach_teacher_scores(self, batch)"))
        self.assertLess(fit.index("attach_teacher_scores(self, batch)"), fit.index("self.actor_rollout_wg.update_actor(batch)"))
        actor = (self.runtime / "verl-agent/verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
        self.assertIn('select_keys.append("teacher_log_probs")', actor)
        self.assertLess(actor.index("policy_loss, sdar_metrics = add_sdar_loss("), actor.index("loss.backward()"))


if __name__ == "__main__":
    unittest.main()
