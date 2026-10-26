"""Verify launch arguments and failure propagation without CUDA or PyTorch."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class RunScriptTest(unittest.TestCase):
    def run_script(self, return_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "scalable_unified_811"
            package.mkdir()
            script = package / "run.sh"
            shutil.copyfile(str(Path(__file__).resolve().parents[1] / "run.sh"), str(script))
            dataset = root / "frames with spaces"
            dataset.mkdir()
            fake_python = root / "fake_python"
            fake_python.write_text(
                '#!/usr/bin/env bash\n'
                'printf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES" > "$CALL_LOG"\n'
                'printf "%s\\n" "$@" >> "$CALL_LOG"\n'
                'exit "$FAKE_EXIT"\n'
            )
            fake_python.chmod(0o755)
            calls = root / "calls.txt"
            env = dict(os.environ, PYTHON_BIN=str(fake_python), CALL_LOG=str(calls),
                       FAKE_EXIT=str(return_code))
            result = subprocess.run(["bash", str(script), "2", str(dataset), "3"],
                                    env=env, capture_output=True, text=True)
            return result, calls.read_text().splitlines(), str(dataset)

    def test_launches_correct_experiment(self):
        result, args, dataset = self.run_script(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[0], "GPU=2")
        self.assertIn("scalable_unified_811.train", args)
        self.assertIn(dataset, args)
        self.assertEqual(args[args.index("--epochs") + 1], "3")
        self.assertEqual(args[args.index("--level_group_sizes") + 1:
                              args.index("--level_group_sizes") + 4], ["8", "1", "1"])
        self.assertEqual(args[args.index("--out_dir") + 1], "./output/scalable_unified_811")

    def test_returns_training_failure(self):
        result, _, _ = self.run_script(7)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("completed", result.stdout)


if __name__ == "__main__":
    unittest.main()
