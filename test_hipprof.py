"""Exercise collection failure handling without GPU/DTK dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fa-bw-hipprof-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).parent
        for name in ("profile_hipprof.sh", "profile_capture.py"):
            if (source / name).exists():
                shutil.copyfile(source / name, self.root / name)
        # Replace GPU-only entry points; keep the shell, capture and checker real.
        (self.root / "test_flash_attn.py").write_text(
            "import unittest\nclass Fixture(unittest.TestCase):\n"
            "    def test_gpu_placeholder(self): pass\n")
        for name in ("diagnose.py", "dump_hipprof.py"):
            (self.root / name).write_text("print('fixture GPU tooling')\n")
        (self.root / "profile_workload.py").write_text('''
import argparse
import os
p = argparse.ArgumentParser()
p.add_argument('--impl')
p.add_argument('--length')
p.add_argument('--warmup', type=int)
p.add_argument('--repeats', type=int)
a = p.parse_args()
print('FA_BW_CXXFLAGS=' + os.environ.get('FA_BW_CXXFLAGS', ''), flush=True)
os.write(2, b'native stderr from workload\\n')
if os.environ.get('INSIDE_FIXTURE_PROFILER') and os.environ['PROFILE_CASE'] == 'child_failure':
    raise RuntimeError('fixture import/runtime failure')
if os.environ.get('INSIDE_FIXTURE_PROFILER') and os.environ['PROFILE_CASE'] == 'abrupt_exit':
    os._exit(9)
if os.environ.get('INSIDE_FIXTURE_PROFILER') and os.environ['PROFILE_CASE'] == 'inner_error':
    os.write(2, b'[ERROR]: profiler error inside application\\n')
print(f'Completed {a.warmup + a.repeats} calls; output shape=(1, 16, 4096, 64)', flush=True)
''')
        bindir = self.root / "bin"
        bindir.mkdir()
        (bindir / "python").symlink_to(sys.executable)
        profiler = bindir / "hipprof"
        profiler.write_text("#!" + sys.executable + "\n" + '''
import os
from pathlib import Path
import subprocess
import sys
args = sys.argv[1:]
if args == ['-h']:
    print('fixture hipprof help')
    raise SystemExit(0)
i = args.index('-o')
prefix = Path(args[i+1])
case = os.environ['PROFILE_CASE']
if case == 'no_launch':
    print("[WARN]: trace.db has not valid trace data.")
    raise SystemExit(0)
# Like the observed failing profiler, hide application stdout/stderr and
# return zero even if the application or export failed.
subprocess.run(args[i+2:], env=dict(os.environ, INSIDE_FIXTURE_PROFILER='1'),
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if case == 'export_failure':
    print('[ERROR]: merge pmc data failed')
if case != 'missing_export':
    suffix = '.hipkernel.csv' if '--hip-trace' in args else '.csv'
    if case == 'api_only':
        suffix = '.hipapi.csv'
    Path(str(prefix) + suffix).write_text('KernelName,Calls\\nattention,3\\n')
''')
        profiler.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"])

    def run_collection(self, case, mode="all", pair="tuned"):
        result = subprocess.run(
            ["bash", str(self.root / "profile_hipprof.sh"), mode, "4096", pair],
            env=dict(self.env, PROFILE_CASE=case), text=True, capture_output=True)
        return result, list((self.root / "build/hipprof").glob("run-*"))

    def test_empty_trace_returning_zero_stops_collection(self):
        result, runs = self.run_collection("no_launch")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Archive:", result.stdout)
        self.assertFalse((runs[0] / "sdpa").exists())

    def test_child_exception_is_visible_despite_hidden_profiler_output(self):
        result, _ = self.run_collection("child_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixture import/runtime failure", result.stdout + result.stderr)
        self.assertIn("native stderr from workload", result.stdout + result.stderr)

    def test_abrupt_child_exit_cannot_look_successful(self):
        result, _ = self.run_collection("abrupt_exit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("status", result.stdout + result.stderr)

    def test_export_error_overrides_successful_workload_and_exit_code(self):
        result, _ = self.run_collection("export_failure", "pmc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("merge pmc data failed", result.stdout + result.stderr)

    def test_missing_export_is_not_a_successful_collection(self):
        result, _ = self.run_collection("missing_export")
        self.assertNotEqual(result.returncode, 0)

    def test_error_in_injected_process_stops_collection(self):
        result, _ = self.run_collection("inner_error")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("profiler error inside application", result.stdout + result.stderr)

    def test_api_csv_does_not_substitute_for_pmc_export(self):
        result, _ = self.run_collection("api_only", "pmc")
        self.assertNotEqual(result.returncode, 0)

    def test_valid_collection_keeps_logs_and_archive(self):
        result, runs = self.run_collection("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Archive:", result.stdout)
        run = next(path for path in runs if path.is_dir())
        for variant in ("tuned", "sdpa"):
            for stage in ("preflight", "trace", "pmc"):
                prefix = run / variant / stage
                self.assertEqual(Path(str(prefix) + ".workload.status").read_text().strip(), "0")
                self.assertIn("native stderr", Path(str(prefix) + ".workload.log").read_text())

    def test_optimized_and_tuned_collect_their_intended_builds(self):
        # The remaining gap must be profiled with the measured implementation,
        # not the older tuned baseline or stale flags from the caller's shell.
        self.env['FA_BW_CXXFLAGS'] = '-DFA_BW_Q_TILE=256 -DFA_BW_KV_TILE=128'
        for pair, enabled in (("optimized", "1"), ("tuned", "0")):
            with self.subTest(pair=pair):
                result, runs = self.run_collection("success", pair=pair)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = next(path for path in runs if path.is_dir() and (path / pair).exists())
                for variant in (pair, "sdpa"):
                    for stage in ("preflight", "trace", "pmc"):
                        content = (run / variant / f"{stage}.workload.log").read_text()
                        line = next(s for s in content.splitlines() if s.startswith('FA_BW_CXXFLAGS='))
                        flags = dict(arg[2:].split('=', 1) for arg in line.split('=', 1)[1].split()
                                     if arg.startswith('-D'))
                        for name, value in {"Q_TILE": "128", "KV_TILE": "64", "MATRIX_V": "1",
                                            "FULL_KV": "1", "ASYNC_K": enabled, "ASYNC_V": enabled,
                                            "DEFER_DENOM": enabled, "NATIVE_EXP2": "1",
                                            "LATE_PREFETCH": "0", "CONDITIONAL_RESCALE": "0"}.items():
                            self.assertEqual(flags.get(f'FA_BW_{name}'), value, content)


if __name__ == "__main__":
    unittest.main()
