"""Test queue order and fail-fast behavior without PyTorch or a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SingleQueueTest(unittest.TestCase):
    def run_queue(self, fail_base=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "scalable_prefix_811"
            package.mkdir()
            source = Path(__file__).resolve().parents[1] / "run_single_queue.sh"
            script = package / source.name
            shutil.copyfile(str(source), str(script))
            data = root / "frames with spaces"
            data.mkdir()
            fake = root / "fake_python"
            fake.write_text(
                '#!/usr/bin/env bash\n'
                'echo "$*" >> "$CALL_LOG"\n'
                'if [[ "${FAIL_BASE:-0}" == 1 && "$*" == *train_base_only* ]]; then exit 7; fi\n'
                'exit 0\n'
            )
            fake.chmod(0o755)
            calls = root / "calls.txt"
            env = dict(os.environ, PYTHON_BIN=str(fake), CALL_LOG=str(calls),
                       FAIL_BASE=str(int(fail_base)))
            result = subprocess.run(["bash", str(script), "2", str(data), "3"],
                                    env=env, capture_output=True, text=True)
            return result, calls.read_text().splitlines()

    def test_base_then_mid(self):
        result, calls = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("train_base_only", calls[0])
        self.assertIn("train_mid_only", calls[1])
        self.assertIn("--epochs 3", calls[0])

    def test_failure_does_not_start_mid(self):
        result, calls = self.run_queue(fail_base=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)
        self.assertIn("queue stopped", result.stderr)


if __name__ == "__main__":
    unittest.main()
