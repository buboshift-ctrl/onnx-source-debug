#!/usr/bin/env python3
"""
코드 트랙 (code track) — 외부 컴파일 경로의 대역(stand-in).

실제 ARX 프로덕션 파이프라인에서 이 자리는:
    ONNX  →  onnx-mlir  →  harx (mlir-arx)  →  EmitC  →  생성된 C 코드
이며, 우리가 건드리지 않는 외부(PNU) 컴파일러다. 이 경로는 ONNX node 의 `metadata_props`/
`doc_string` 을 버리고 `node.name` 흔적만 남긴 채 C 를 뱉는다.

이 컨테이너에는 실제 harx/EmitC 가 없으므로, 아래 `get_generated_c()` 가 **데모 C 생성기로
대체돼 있다.** ONNX 그래프에서 `node.name` 과 `op_type` 만 읽어(정보 트랙이 쓰는 provenance 는
일절 읽지 않는다), 노드마다 `/* onnx_node_name: <name> */` 마커가 붙은 작은 커널 함수를 만든다.

────────────────────────────────────────────────────────────────────────────
실제 harx/EmitC 연동 시:
  * `get_generated_c(onnx_path: str) -> str` 의 **구현부만 교체**하면 된다.
    - 입력: onnx 파일 경로
    - 출력: C 소스 문자열 (노드 구획마다 `onnx_node_name` 이 식별 가능한 형태로 들어있어야 함.
            join.py 의 마커 정규식 `onnx[_-]node(_name)? [:=] <name>` 에 걸리면 됨 —
            주석이든 pragma 든 형식은 관대하게 인식)
  * 시그니처(경로 → C 문자열)는 유지할 것. join.py / run_full_pipeline.py 는 그대로 둔다.
  * 정보 트랙과 독립: 이 함수는 locmap 도, 원본 .py 위치도 몰라야 한다.
────────────────────────────────────────────────────────────────────────────

공개 API:
  get_generated_c(onnx_path: str) -> str

결정성: 그래프 순서 그대로, 타임스탬프/난수 없음 → 같은 .onnx 면 바이트 동일 C.
"""
from __future__ import annotations

import re

# op_type → C 함수 이름에 쓸 짧은 별칭 (데모용 주석에만 사용)
_C_TYPE_BY_OP = {
    "MaxPool": "maxpool", "Gemm": "gemm", "Relu": "relu",
    "Reshape": "reshape", "Conv": "conv", "Add": "add", "MatMul": "matmul",
}


def _fn_name(node_name: str) -> str:
    return f"node_{re.sub(r'[^A-Za-z0-9_]', '_', node_name)}"


def _synth_demo_c(graph_nodes: list[tuple[str, str]]) -> str:
    """
    graph_nodes: [(node_name, op_type), ...] (그래프 순서).
    EmitC 가 낼 법한 모양의 데모 C 를 만든다. 노드 구획은 정확히 7줄:
        static void <fn>(...) {          /* 시그니처 — C 에 매핑 */
        /* onnx_node_name: <name> ... */  /* 마커 — 이 다음에 join 이 #line 삽입 */
          for (...)                       /* 연산 본문 — 원본 .py 로 매핑 */
            buf[i] = ...;
        /* onnx_node_end */               /* 구획 끝 — 여기서 C 로 되돌림 */
        }
        <blank>
    (이 7줄 구조는 join.py 의 #line 계산이 의존하므로 바꾸지 말 것.)
    """
    L: list[str] = [
        "#include <stddef.h>",
        "",
        "/* stand-in for EmitC-generated kernels; one function per ONNX node. */",
        "",
    ]
    call_seq: list[str] = []
    for name, op in graph_nodes:
        kind = _C_TYPE_BY_OP.get(op, "op")
        fn = _fn_name(name)
        call_seq.append(f"  {fn}(buf, n);")
        L += [
            f"static void {fn}(float *buf, size_t n) {{   /* {kind} stand-in */",
            f"/* onnx_node_name: {name}   op={op} */",
            f"  for (size_t i = 0; i < n; ++i)",
            f"    buf[i] = buf[i] * 1.0f + 0.0f;   /* {op} */",
            "/* onnx_node_end */",
            "}",
            "",
        ]

    L += [
        "void main_graph(float *buf, size_t n) {",
        *call_seq,
        "}",
        "",
        "int main(void) {",
        "  float buf[8] = {0};",
        "  main_graph(buf, 8);",
        "  return (int)buf[0];",
        "}",
        "",
    ]
    return "\n".join(L) + "\n"


def get_generated_c(onnx_path: str) -> str:
    """
    ONNX 파일 → 생성된 C 소스 문자열.

    현재 구현: 데모 생성기. ONNX 그래프에서 (node.name, op_type) 만 읽어 마커가 붙은
    커널 C 를 만든다. 실제 harx/EmitC 연동 시 이 함수 본문만 교체(모듈 docstring 참고).
    """
    import onnx

    model = onnx.load(onnx_path)
    graph_nodes: list[tuple[str, str]] = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"__anon_{idx}__{node.op_type}"
        graph_nodes.append((name, node.op_type))
    return _synth_demo_c(graph_nodes)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="코드 트랙 대역 — ONNX → 데모 C")
    ap.add_argument("onnx", help=".onnx 파일")
    ap.add_argument("-o", "--output", help="출력 C 경로 (생략 시 stdout)")
    args = ap.parse_args(argv)

    c = get_generated_c(args.onnx)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(c)
        print(f"C 기록: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
