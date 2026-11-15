"""Classify square patterns using two filters and explicit MAC loops."""

import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter


EPSILON = 1e-9
BENCHMARK_REPEATS = 1000
MATRIX_SIZES = (3, 5, 13, 25)
JSON_SIZES = (5, 13, 25)
DEFAULT_DATA = Path(__file__).resolve().with_name("data.json")
CASE_KEY = re.compile(r"size_(5|13|25)_[0-9]+")
FILTER_KEYS = {"Cross": "cross", "X": "x"}
EXPECTED_LABELS = {"+": "Cross", "x": "X"}
PATTERN_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def validate_matrix(matrix, size, label):
    """Require a numeric N×N matrix before passing it to mac()."""
    if not isinstance(matrix, list):
        raise ValueError(f"{label}: 2차원 배열이 필요합니다.")
    if len(matrix) != size:
        raise ValueError(
            f"크기 불일치: {label}의 행은 {size}개여야 합니다 "
            f"(실제 {len(matrix)}개)."
        )
    for row_number, row in enumerate(matrix, start=1):
        if not isinstance(row, list):
            raise ValueError(f"{label}의 {row_number}행은 배열이어야 합니다.")
        if len(row) != size:
            raise ValueError(
                f"크기 불일치: {label}의 {row_number}행 열은 {size}개여야 합니다 "
                f"(실제 {len(row)}개)."
            )
        for column_number, value in enumerate(row, start=1):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(
                    f"{label}의 {row_number}행 {column_number}열에는 "
                    "유한한 숫자가 필요합니다."
                )


def mac(pattern, filter_matrix):
    """Return one MAC score. Inputs must have been validated as equal squares."""
    score = 0
    size = len(pattern)
    for row in range(size):
        for col in range(size):
            score += pattern[row][col] * filter_matrix[row][col]
    return score


def compare_scores(score_a, score_b, epsilon=EPSILON):
    """Classify two scores using a strict absolute difference."""
    if not (math.isfinite(score_a) and math.isfinite(score_b)):
        raise ValueError("MAC 점수가 유한한 숫자 범위를 벗어났습니다.")
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"
    return "A" if score_a > score_b else "B"


def read_matrix_3(label):
    print(f"{label} 입력 (3줄, 한 줄에 숫자 3개):")
    matrix = []
    for row_number in range(1, 4):
        while True:
            try:
                line = input(f"  {row_number}행 > ")
            except EOFError as exc:
                raise EOFError(
                    f"입력 형식 오류: {label}은 3행이 필요합니다. "
                    f"{len(matrix)}행까지 받았습니다. 다시 실행해 입력하세요."
                ) from exc
            try:
                row = [float(token) for token in line.split()]
                if len(row) != 3 or not all(math.isfinite(value) for value in row):
                    raise ValueError
            except ValueError:
                print("입력 형식 오류: 숫자 3개를 공백으로 구분해 같은 행을 다시 입력하세요.")
                continue
            matrix.append(row)
            break
    return matrix


def benchmark_input(size):
    # Use deterministic, prebuilt inputs even when a JSON case is malformed.
    if size == 3:
        return PATTERN_3, PATTERN_3
    center = size // 2
    cross = [[int(i == center or j == center) for j in range(size)] for i in range(size)]
    diagonal = [[int(i == j or i + j == size - 1) for j in range(size)] for i in range(size)]
    return cross, diagonal


def benchmark_mac(sizes, repeats=BENCHMARK_REPEATS):
    print(f"\n성능 분석 (패턴 1개, 필터 1개) | 반복: {repeats}회/크기")
    print("크기  | 평균 시간(ms) | MAC 횟수(N²)")
    for size in sizes:
        pattern, filter_matrix = benchmark_input(size)
        start = perf_counter()
        for _ in range(repeats):
            mac(pattern, filter_matrix)
        average_ms = (perf_counter() - start) * 1000 / repeats
        print(f"{size}×{size} | {average_ms:.6f} | {size * size}")


def run_interactive():
    filter_a = read_matrix_3("필터 A")
    filter_b = read_matrix_3("필터 B")
    print("필터 A·B 저장 완료 (각 3×3)")
    pattern = read_matrix_3("패턴")

    start = perf_counter()
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed_ms = (perf_counter() - start) * 1000
    prediction = compare_scores(score_a, score_b)
    shown_prediction = "판정 불가 (UNDECIDED)" if prediction == "UNDECIDED" else prediction
    print(f"\n점수: A={score_a:.12g} | B={score_b:.12g}")
    print(f"두 필터 MAC 계산 시간: {elapsed_ms:.6f} ms")
    print(f"판정: {shown_prediction}")
    benchmark_mac((3,))


def load_data(path):
    with path.open(encoding="utf-8-sig") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")
    for key in ("filters", "patterns"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"JSON의 {key} 값은 객체여야 합니다.")
    return data["filters"], data["patterns"]


def prepare_filters(filters):
    by_size = {}
    errors = {}
    for size in JSON_SIZES:
        group_name = f"size_{size}"
        try:
            group = filters.get(group_name)
            if not isinstance(group, dict):
                raise ValueError(f"필터 묶음 filters.{group_name}이 없거나 객체가 아닙니다.")
            normalized = {}
            for label, raw_key in FILTER_KEYS.items():
                matrix = group.get(raw_key)
                validate_matrix(matrix, size, f"{group_name}.{raw_key} 필터")
                normalized[label] = matrix
        except (ValueError, OverflowError) as exc:
            errors[size] = str(exc)
        else:
            by_size[size] = normalized
    return by_size, errors


def evaluate_case(case_id, case, by_size, filter_errors):
    key = CASE_KEY.fullmatch(case_id)
    if not key:
        raise ValueError("키 형식 오류: size_{5|13|25}_{숫자 인덱스}가 필요합니다.")
    size = int(key.group(1))
    if size in filter_errors:
        raise ValueError(filter_errors[size])
    if not isinstance(case, dict):
        raise ValueError("케이스 값은 객체여야 합니다.")
    pattern = case.get("input")
    validate_matrix(pattern, size, "입력 패턴")
    raw_expected = case.get("expected")
    if not isinstance(raw_expected, str) or raw_expected not in EXPECTED_LABELS:
        raise ValueError("expected 값은 '+' 또는 'x'여야 합니다.")
    expected = EXPECTED_LABELS[raw_expected]

    filters = by_size[size]
    score_cross = mac(pattern, filters["Cross"])
    score_x = mac(pattern, filters["X"])
    winner = compare_scores(score_cross, score_x)
    prediction = {"A": "Cross", "B": "X", "UNDECIDED": "UNDECIDED"}[winner]
    return score_cross, score_x, prediction, expected


def run_json(path):
    filters, patterns = load_data(path)
    by_size, filter_errors = prepare_filters(filters)
    print(f"\n데이터 파일: {path.name}")
    for size, reason in filter_errors.items():
        print(f"필터 검증 오류 (size_{size}): {reason}")

    failures = []
    for case_id, case in patterns.items():
        try:
            cross, x, prediction, expected = evaluate_case(
                case_id, case, by_size, filter_errors
            )
        except (ValueError, OverflowError) as exc:
            print(f"{case_id} | Cross=N/A | X=N/A | 판정: N/A | FAIL | {exc}")
            failures.append((case_id, str(exc)))
            continue

        status = "PASS" if prediction == expected else "FAIL"
        print(
            f"{case_id} | Cross={cross:.12g} | X={x:.12g} | "
            f"차이={abs(cross - x):.12g} | 판정: {prediction} | "
            f"expected: {expected} | {status}"
        )
        if status == "FAIL":
            failures.append((case_id, f"기대 라벨 불일치 (판정 {prediction}, 정답 {expected})"))

    benchmark_mac(MATRIX_SIZES)
    print(f"\n전체: {len(patterns)} | PASS: {len(patterns) - len(failures)} | FAIL: {len(failures)}")
    if not failures:
        print("실패 사례: 없음")
    else:
        print("실패 사례:")
        for case_id, reason in failures:
            print(f"  - {case_id}: {reason}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini NPU MAC 패턴 분류기")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="모드 2에 사용할 JSON 경로")
    args = parser.parse_args(argv)
    print("Mini NPU 시뮬레이터")
    print("1. 사용자 입력 (3×3)\n2. JSON 데이터 분석")
    try:
        while True:
            choice = input("모드 선택 > ").strip()
            if choice in {"1", "2"}:
                break
            print("1 또는 2를 입력하세요.")

        if choice == "1":
            run_interactive()
        else:
            run_json(args.data)
    except EOFError as exc:
        print(f"\n{exc or '입력이 종료되었습니다. 다시 실행해 모드를 선택하세요.'}")
        return 1
    except (OSError, ValueError, OverflowError) as exc:
        print(f"\n실행 오류: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n실행을 취소했습니다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
