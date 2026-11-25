"""Mini NPU 검증. 저장소 루트에서 python tests/check_cases.py로 실행한다."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_npu import compare_scores, mac, normalize_label, parse_case_size


CROSS = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
ZERO = [[0, 0, 0] for _ in range(3)]


def matrix_lines(matrix):
    return [" ".join(str(value) for value in row) for row in matrix]


class ScoreTests(unittest.TestCase):
    def test_normalize_filter_and_expected_labels(self):
        for source, raw, expected in (
            ("filter", "cross", "Cross"),
            ("filter", "x", "X"),
            ("expected", "+", "Cross"),
            ("expected", "x", "X"),
        ):
            with self.subTest(source=source, raw=raw):
                self.assertEqual(normalize_label(raw, source), expected)
        for source, raw in (("filter", "+"), ("expected", "cross"), ("expected", [])):
            with self.subTest(source=source, raw=raw):
                with self.assertRaises(ValueError):
                    normalize_label(raw, source)

    def test_case_size_checks_format_and_supported_sizes(self):
        self.assertEqual(parse_case_size("size_13_2"), 13)
        with self.assertRaisesRegex(ValueError, "키 형식 오류"):
            parse_case_size("size_5_cross")
        with self.assertRaisesRegex(ValueError, "지원하지 않는 크기"):
            parse_case_size("size_1000_1")

    def test_mac_matches_hand_calculation(self):
        self.assertEqual(mac(CROSS, CROSS), 5)
        self.assertEqual(mac(CROSS, X), 1)
        # This asymmetric example distinguishes matching positions from transposition.
        pattern = [[1, -2, 3], [0.5, 0, 0], [0, 0, 0]]
        filter_matrix = [[0, 4, 2], [-2, 0, 0], [0, 0, 0]]
        self.assertEqual(mac(pattern, filter_matrix), -3)

    def test_epsilon_and_strict_boundary(self):
        epsilon = 1e-9
        for left, right, expected in (
            (5, 1, "A"),
            (1, 5, "B"),
            (0, 0, "UNDECIDED"),
            (0, math.nextafter(epsilon, 0.0), "UNDECIDED"),
            (0, epsilon, "B"),
            (0, math.nextafter(epsilon, math.inf), "B"),
            (epsilon, 0, "A"),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(compare_scores(left, right, epsilon), expected)


class ConsoleTests(unittest.TestCase):
    def run_mode(self, lines, *arguments):
        result = subprocess.run(
            [sys.executable, str(ROOT / "mini_npu.py"), *arguments],
            input="\n".join(lines) + "\n",
            cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def assert_benchmark(self, output, sizes):
        repetitions = re.search(r"반복:\s*(\d+)", output)
        self.assertIsNotNone(repetitions, output)
        self.assertGreaterEqual(int(repetitions.group(1)), 10)
        self.assertIn("ms", output)
        self.assertIn("N²", output)
        for n in sizes:
            self.assertRegex(
                output,
                rf"(?m)^\s*{n}×{n}\s*\|\s*[0-9]+(?:\.[0-9]+)?\s*\|\s*{n*n}\s*$",
            )

    def test_manual_mode_retries_and_classifies(self):
        output = self.run_mode(
            [
                "1", "1 0", "1 a 0",
                *matrix_lines(CROSS), *matrix_lines(X), *matrix_lines(CROSS),
            ]
        )
        self.assertEqual(output.count("입력 형식 오류"), 2)
        self.assertIn("A=5", output)
        self.assertIn("B=1", output)
        self.assertIn("판정: A", output)
        for label in ("필터 A", "필터 B", "두 필터"):
            self.assertRegex(output, rf"{label} MAC 계산 시간: [0-9.]+ ms")
        self.assertIn("예: 0 1 0", output)
        self.assert_benchmark(output, [3])

    def test_manual_mode_zero_pattern_is_undecided(self):
        output = self.run_mode(["1", *matrix_lines(CROSS), *matrix_lines(X), *matrix_lines(ZERO)])
        self.assertIn("A=0", output)
        self.assertIn("B=0", output)
        self.assertIn("판정 불가 (UNDECIDED)", output)

    def test_epsilon_and_repeats_apply_in_manual_mode(self):
        output = self.run_mode(
            ["1", *matrix_lines(CROSS), *matrix_lines(X), *matrix_lines(CROSS)],
            "--epsilon", "5", "--repeats", "20",
        )
        self.assertIn("판정 불가 (UNDECIDED)", output)
        self.assertIn("반복: 20회/크기", output)

    def test_json_ties_are_counted_and_saved_with_reasons(self):
        with TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            output = self.run_mode(
                ["2"], "--epsilon", "9", "--repeats", "20",
                "--summary-json", str(summary_path),
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("반복: 20회/크기", output)
        self.assertIn("전체: 6 | PASS: 4 | FAIL: 2", output)
        self.assertIn("size_5_1: 동점 (UNDECIDED)", output)
        self.assertEqual((summary["total"], summary["passed"], summary["failed"]), (6, 4, 2))
        self.assertEqual(
            [failure["case_id"] for failure in summary["failures"]], ["size_5_1", "size_5_2"]
        )
        self.assertTrue(all("동점 (UNDECIDED)" in failure["reason"] for failure in summary["failures"]))

    def test_json_load_error_shows_path_and_expected_structure(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "mini_npu.py"), "--data", "missing-data.json"],
            input="2\n", cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True, encoding="utf-8", timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 1)
        for message in ("요청 경로:", "missing-data.json", "기본 경로:", "filters.size_5.cross"):
            self.assertIn(message, result.stdout)

    def test_bundled_sample_passes_and_reports_all_sizes(self):
        output = self.run_mode(["2"])
        cases = re.findall(r"(?m)^size_(5|13|25)_\d+ \| .*\bPASS\b.*$", output)
        self.assertCountEqual(cases, ["5", "5", "13", "13", "25", "25"])
        self.assertIn("전체: 6 | PASS: 6 | FAIL: 0", output)
        self.assert_benchmark(output, [3, 5, 13, 25])

    def test_bad_size_is_reported_and_next_case_runs(self):
        fixture = ROOT / "tests" / "fixtures" / "invalid_size.json"
        output = self.run_mode(["2"], "--data", str(fixture))
        bad_case = re.search(r"(?m)^size_5_1 \| .*FAIL.*$", output)
        next_case = re.search(r"(?m)^size_5_2 \| .*PASS.*$", output)
        self.assertIsNotNone(bad_case, output)
        self.assertIsNotNone(next_case, output)
        self.assertIn("크기 불일치", bad_case.group())
        self.assertLess(bad_case.start(), next_case.start())
        self.assertIn("전체: 2 | PASS: 1 | FAIL: 1", output)
        _, heading, failures = output.partition("실패 사례")
        self.assertTrue(heading, output)
        self.assertIn("size_5_1", failures)
        self.assertIn("크기 불일치", failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
