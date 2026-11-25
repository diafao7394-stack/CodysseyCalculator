"""입력 패턴과 두 필터의 MAC 점수를 비교하는 콘솔 프로그램."""

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
CASE_KEY = re.compile(r"size_([1-9][0-9]*)_[0-9]+")
# 같은 모양의 새 기호는 해당 매핑에 별칭으로 추가한다.
# 새로운 모양을 지원하려면 점수를 계산하고 승자를 고르는 로직도 확장해야 한다.
FILTER_LABELS = {"cross": "Cross", "x": "X"}
EXPECTED_LABELS = {"+": "Cross", "x": "X"}
PATTERN_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def validate_matrix(matrix, size, label):
    """각 행과 열의 개수, 각 칸의 숫자를 검사한다."""
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
    """크기가 같은 두 배열에서 같은 위치의 값을 곱해 모두 더한다."""
    # 점수 = Σ (pattern[row][col] × filter_matrix[row][col])
    score = 0
    size = len(pattern)
    for row in range(size):
        for col in range(size):
            score += pattern[row][col] * filter_matrix[row][col]
    return score


def compare_scores(score_a, score_b, epsilon=EPSILON):
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
                    f"{len(matrix)}행까지 받았습니다. 행 예시: 0 1 0. 다시 실행해 입력하세요."
                ) from exc
            try:
                row = [float(token) for token in line.split()]
                if len(row) != 3 or not all(math.isfinite(value) for value in row):
                    raise ValueError
            except ValueError:
                print(
                    "입력 형식 오류: 3×3 배열은 한 행에 숫자 3개가 필요합니다. "
                    "공백으로 구분해 같은 행을 다시 입력하세요. 예: 0 1 0"
                )
                continue
            matrix.append(row)
            break
    return matrix


def benchmark_input(size):
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
        # 배열을 만드는 시간이 MAC 측정에 섞이지 않도록 먼저 준비한다.
        pattern, filter_matrix = benchmark_input(size)
        # 측정 구간: mac() 반복 호출부터 마지막 호출 직후까지.
        start = perf_counter()
        for _ in range(repeats):
            mac(pattern, filter_matrix)
        average_ms = (perf_counter() - start) * 1000 / repeats
        print(f"{size}×{size} | {average_ms:.6f} | {size * size}")


def run_interactive(epsilon, repeats):
    filter_a = read_matrix_3("필터 A")
    filter_b = read_matrix_3("필터 B")
    print("필터 A·B 저장 완료 (각 3×3)")
    pattern = read_matrix_3("패턴")

    start = perf_counter()
    score_a = mac(pattern, filter_a)
    after_a = perf_counter()
    score_b = mac(pattern, filter_b)
    after_b = perf_counter()
    prediction = compare_scores(score_a, score_b, epsilon)
    shown_prediction = "판정 불가 (UNDECIDED)" if prediction == "UNDECIDED" else prediction
    print(f"\n점수: A={score_a:.12g} | B={score_b:.12g}")
    print(f"필터 A MAC 계산 시간: {(after_a - start) * 1000:.6f} ms")
    print(f"필터 B MAC 계산 시간: {(after_b - after_a) * 1000:.6f} ms")
    print(f"두 필터 MAC 계산 시간: {(after_b - start) * 1000:.6f} ms")
    print(f"판정: {shown_prediction}")
    benchmark_mac((3,), repeats)


def load_data(path):
    try:
        with path.open(encoding="utf-8-sig") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError("JSON 최상위 값은 객체여야 합니다.")
        for key in ("filters", "patterns"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"JSON의 {key} 값은 객체여야 합니다.")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"JSON을 읽을 수 없습니다. 요청 경로: {path.absolute()}\n"
            f"기본 경로: {DEFAULT_DATA}\n"
            f"원인: {exc}\n"
            'UTF-8 JSON 파일을 확인하거나 --data로 경로를 지정하세요. '
            '구조 예: filters.size_5.cross, filters.size_5.x, '
            'patterns.size_5_1의 input(5×5 배열)과 expected("+" 또는 "x").'
        ) from exc
    return data["filters"], data["patterns"]


def normalize_label(label, source):
    """필터 키와 expected 값을 점수 비교에 쓰는 Cross 또는 X로 바꾼다."""
    labels = {"filter": FILTER_LABELS, "expected": EXPECTED_LABELS}[source]
    if not isinstance(label, str) or label not in labels:
        allowed = ", ".join(repr(value) for value in labels)
        raise ValueError(f"{source} 라벨은 {allowed} 중 하나여야 합니다.")
    return labels[label]


def prepare_filters(filters):
    filters_by_size = {}
    filter_errors = {}
    for size in JSON_SIZES:
        group_name = f"size_{size}"
        try:
            group = filters.get(group_name)
            if not isinstance(group, dict):
                raise ValueError(f"필터 묶음 filters.{group_name}이 없거나 객체가 아닙니다.")
            normalized = {}
            for raw_key in FILTER_LABELS:
                if raw_key not in group:
                    continue
                label = normalize_label(raw_key, "filter")
                if label in normalized:
                    raise ValueError(f"{group_name}: {label} 필터가 중복되었습니다.")
                matrix = group[raw_key]
                validate_matrix(matrix, size, f"{group_name}.{raw_key} 필터")
                normalized[label] = matrix
            for label in ("Cross", "X"):
                if label not in normalized:
                    raise ValueError(f"{group_name}: {label} 필터가 필요합니다.")
        except (ValueError, OverflowError) as exc:
            filter_errors[size] = str(exc)
        else:
            filters_by_size[size] = normalized
    return filters_by_size, filter_errors


def parse_case_size(case_id):
    match = CASE_KEY.fullmatch(case_id)
    if not match:
        raise ValueError("키 형식 오류: size_5_1처럼 size_{크기}_{숫자 인덱스}가 필요합니다.")
    size = int(match.group(1))
    if size not in JSON_SIZES:
        sizes = ", ".join(map(str, JSON_SIZES))
        raise ValueError(f"지원하지 않는 크기 {size}. 사용 가능한 크기: {sizes}")
    return size


def read_case(case_id, case, filter_errors):
    size = parse_case_size(case_id)
    if size in filter_errors:
        raise ValueError(filter_errors[size])
    if not isinstance(case, dict):
        raise ValueError("케이스 값은 객체여야 합니다.")
    pattern = case.get("input")
    validate_matrix(pattern, size, "입력 패턴")
    return size, pattern, normalize_label(case.get("expected"), "expected")


def evaluate_case(case_id, case, filters_by_size, filter_errors, epsilon=EPSILON):
    size, pattern, expected = read_case(case_id, case, filter_errors)
    filters = filters_by_size[size]
    score_cross = mac(pattern, filters["Cross"])
    score_x = mac(pattern, filters["X"])
    winner = compare_scores(score_cross, score_x, epsilon)
    prediction = {"A": "Cross", "B": "X", "UNDECIDED": "UNDECIDED"}[winner]
    return score_cross, score_x, prediction, expected


def run_json(path, epsilon, repeats, summary_path):
    filters, patterns = load_data(path)
    filters_by_size, filter_errors = prepare_filters(filters)
    print(f"\n데이터 파일: {path.name}")
    for size, reason in filter_errors.items():
        print(f"필터 검증 오류 (size_{size}): {reason}")

    failures = []
    for case_id, case in patterns.items():
        try:
            cross, x, prediction, expected = evaluate_case(
                case_id, case, filters_by_size, filter_errors, epsilon
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
        if prediction == "UNDECIDED":
            failures.append((case_id, f"동점 (UNDECIDED): 점수 차이 < {epsilon:g}, 정답 {expected}"))
        elif status == "FAIL":
            failures.append((case_id, f"기대 라벨 불일치 (판정 {prediction}, 정답 {expected})"))

    benchmark_mac(MATRIX_SIZES, repeats)
    passed = len(patterns) - len(failures)
    print(f"\n전체: {len(patterns)} | PASS: {passed} | FAIL: {len(failures)}")
    if not failures:
        print("실패 사례: 없음")
    else:
        print("실패 사례:")
        for case_id, reason in failures:
            print(f"  - {case_id}: {reason}")

    if summary_path is not None:
        summary = {
            "total": len(patterns),
            "passed": passed,
            "failed": len(failures),
            "failures": [{"case_id": case_id, "reason": reason} for case_id, reason in failures],
        }
        with summary_path.open("x", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"요약 저장: {summary_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini NPU MAC 패턴 분류기")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="모드 2에 사용할 JSON 경로")
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="동점 허용오차 (기본 1e-9)")
    parser.add_argument(
        "--repeats", type=int, default=BENCHMARK_REPEATS,
        help="성능 측정 크기별 반복 횟수 (기본 1000, 최소 10)",
    )
    parser.add_argument("--summary-json", type=Path, help="모드 2의 요약을 저장할 새 JSON 파일")
    args = parser.parse_args(argv)
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon은 0보다 큰 유한한 숫자여야 합니다.")
    if args.repeats < 10:
        parser.error("--repeats는 10 이상이어야 합니다.")
    print("Mini NPU 시뮬레이터")
    print("1. 사용자 입력 (3×3)\n2. JSON 데이터 분석")
    try:
        while True:
            choice = input("모드 선택 > ").strip()
            if choice in {"1", "2"}:
                break
            print("1 또는 2를 입력하세요.")

        if choice == "1":
            if args.summary_json is not None:
                parser.error("--summary-json은 모드 2에서 사용할 수 있습니다.")
            run_interactive(args.epsilon, args.repeats)
        else:
            run_json(args.data, args.epsilon, args.repeats, args.summary_json)
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
