"""Check the isolated launch namespace without using PyTorch or a GPU."""

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
            package = root / "scalable_unified_additive_811"
            package.mkdir()
            script = package / "run.sh"
            shutil.copyfile(str(Path(__file__).resolve().parents[1] / "run.sh"), str(script))
            dataset = root / "frames with spaces"
            dataset.mkdir()
            fake = root / "fake_python"
            fake.write_text('#!/usr/bin/env bash\n'
                            'printf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES" > "$CALL_LOG"\n'
                            'printf "%s\\n" "$@" >> "$CALL_LOG"\n'
                            'exit "$FAKE_EXIT"\n')
            fake.chmod(0o755)
            calls = root / "calls.txt"
            env = dict(os.environ, PYTHON_BIN=str(fake), CALL_LOG=str(calls), FAKE_EXIT=str(return_code))
            result = subprocess.run(["bash", str(script), "2", str(dataset), "3"],
                                    env=env, capture_output=True, text=True)
            return result, calls.read_text().splitlines(), str(dataset)

    def test_launches_additive_811(self):
        result, args, dataset = self.run_script(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[0], "GPU=2")
        self.assertIn(dataset, args)
        self.assertIn("scalable_unified_additive_811.train", args)
        self.assertEqual(args[args.index("--epochs") + 1], "3")
        self.assertEqual(args[args.index("--hidden_dim") + 1], "256")
        self.assertNotIn("--num_blocks", args)
        self.assertNotIn("--feature_dim", args)
        self.assertEqual(args[args.index("--out_dir") + 1], "./output/scalable_unified_additive_811")

    def test_propagates_failure(self):
        result, _, _ = self.run_script(7)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("completed", result.stdout)


if __name__ == "__main__":
    unittest.main()
