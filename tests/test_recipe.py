import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
import run_alignment


class RecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = reproduce.verify_source(Path(os.environ.get("SEENAV_TEST_SOURCE", reproduce.DEFAULT_SOURCE)))
        cls.native = (cls.source / reproduce.LOCK["train_launcher"]).read_text(encoding="utf-8")

    def test_four_card_is_exactly_one_replacement(self):
        adapted = reproduce.four_gpu_launcher(self.native)
        changes = [line for line in difflib.ndiff(self.native.splitlines(), adapted.splitlines())
                   if line.startswith(("- ", "+ "))]
        self.assertEqual(changes, [
            "-     trainer.n_gpus_per_node=8 \\",
            "+     trainer.n_gpus_per_node=4 \\",
        ])
        self.assertEqual(adapted.replace(reproduce.NEW_GPU, reproduce.OLD_GPU, 1), self.native)

    def test_routed_paths_do_not_modify_recipe(self):
        adapted = reproduce.four_gpu_launcher(self.native)
        launch = reproduce.isolated_paths(adapted)
        removed = [line[2:].strip() for line in difflib.ndiff(adapted.splitlines(), launch.splitlines())
                   if line.startswith("- ")]
        self.assertEqual(set(removed), {
            "--val_data_size $val_data_size",
            "data.train_files=$HOME/data/verl-agent/visual/train.parquet " + chr(92),
            "data.val_files=$HOME/data/verl-agent/visual/test.parquet " + chr(92),
            "data.metrics_save_dir=your_metrics_save_path " + chr(92),
        })
        self.assertIn('trainer.default_local_dir="${SEENAV_RUN_DIR}/checkpoints"', launch)
        self.assertIn('data.metrics_save_dir="${SEENAV_RUN_DIR}/metrics"', launch)
        self.assertIn('--local_dir "${SEENAV_RUN_DIR}/data"', launch)
        for setting in ("algorithm.adv_estimator=srgpo", "algorithm.srgpo.step_group_size=16",
                        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
                        "trainer.save_freq=150", "trainer.test_freq=5", "trainer.total_epochs=150",
                        "trainer.logger=['console','wandb']", "actor_rollout_ref.actor.kl_loss_coef=0.01",
                        "actor_rollout_ref.ref.fsdp_config.param_offload=True", "group_size=4"):
            self.assertIn(setting, launch)

    def test_upstream_drift_rejected(self):
        for bad in ("", self.native + self.native):
            with self.assertRaises(ValueError):
                reproduce.four_gpu_launcher(bad)

    def test_shell_passes_only_card_path_model_tag_and_opsd_differences_to_native_modules(self):
        bash = os.environ.get("SEENAV_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("Bash is not installed")

        def capture(script):
            # Capture Bash's actual argv without importing or running the GPU stack.
            stub = "python3() { printf '%s\\0' \"$@\" __CALL_END__; }\n"
            out = subprocess.check_output([bash, "--noprofile", "--norc", "-e", "-c", stub + script],
                env={**os.environ, "SEENAV_RUN_DIR": "/tmp/seenav-fixture",
                     "SEENAV_BASE_MODEL": "/tmp/pinned-qwen"},
                stderr=subprocess.PIPE)
            calls = [call.split("\0")[:-1] for call in out.decode().split("__CALL_END__\0")[:-1]]
            self.assertEqual(len(calls), 2)
            return calls

        old_data, old_train = capture(self.native)
        new_data, new_train = capture(run_alignment.effective_launcher(self.native))
        self.assertEqual(new_data, old_data + ["--local_dir", "/tmp/seenav-fixture/data"])
        self.assertEqual(old_train[:2], new_train[:2])
        self.assertEqual(new_train[:2], ["-m", "verl.trainer.main_ppo"])
        old_args = dict(arg.split("=", 1) for arg in old_train[2:])
        new_args = dict(arg.split("=", 1) for arg in new_train[2:])
        old_args.update({
            "trainer.n_gpus_per_node": "4",
            "actor_rollout_ref.actor.output_format": "tags",
            "actor_rollout_ref.model.path": "/tmp/pinned-qwen",
            "data.train_files": "/tmp/seenav-fixture/data/visual/train.parquet",
            "data.val_files": "/tmp/seenav-fixture/data/visual/test.parquet",
            "data.metrics_save_dir": "/tmp/seenav-fixture/metrics",
            "trainer.default_local_dir": "/tmp/seenav-fixture/checkpoints",
            "++actor_rollout_ref.actor.use_sdar_loss": "True",
            "++actor_rollout_ref.actor.sdar_loss_coef": "0.01",
            "++actor_rollout_ref.actor.sdar_gate_beta": "5.0",
        })
        self.assertEqual(old_args, new_args)
        self.assertNotIn("++actor_rollout_ref.actor.rethink_local_hints", new_args)

    def test_respect_device_allocation(self):
        self.assertEqual(reproduce.gpu_ids(None, "2,3,6,7"), ["2", "3", "6", "7"])
        self.assertEqual(reproduce.gpu_ids("1,4,5,6", None), ["1", "4", "5", "6"])
        for requested, visible in [(None, None), ("0,1,2,2", None), ("0,1,2,3", "1,2,3,4")]:
            with self.assertRaises(ValueError):
                reproduce.gpu_ids(requested, visible)


if __name__ == "__main__":
    unittest.main()
