#!/usr/bin/env python3
"""
오케스트레이터 (TASK_formalize_architecture_v3.md).

    locmap    = info_track.build_locmap(onnx_path)      # 정보 트랙 (독립)
    c_src     = code_track.get_generated_c(onnx_path)   # 코드 트랙 (독립, 현재는 데모)
    annotated = join.inject(c_src, locmap)              # 유일한 만남 (조인 키 = node.name)
    → gcc -g → readelf/gdb 검증 → 3회 반복 재현성 체크

이 스크립트는 "새 기술 검증"이 아니라 v2 데모(inject_line_directives.py --demo)가 하던 일을
정리한 것이다. 검증 결과는 v2 와 동일해야 한다(회귀 없음).

사용:
    python3 run_full_pipeline.py [ONNX]  [--workdir DIR] [--relative-to DIR]
    (ONNX 생략 시 artifacts/tinycnn_dynamo.onnx)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import code_track
import info_track
import join

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------- #
# 검증 (gcc -g → readelf / gdb).  v2 inject_line_directives.verify_dwarf 와 동일 로직.
# --------------------------------------------------------------------------- #
def verify_dwarf(binary: str, locmap: dict, matched: list[str]) -> dict:
    result: dict = {"checks": [], "ok": True}

    def check(desc: str, cond: bool, detail: str = "") -> None:
        result["checks"].append({"desc": desc, "ok": bool(cond), "detail": detail})
        if not cond:
            result["ok"] = False

    rd = _run(["readelf", "--debug-dump=decodedline", binary])
    decoded = rd.stdout

    want: set[tuple[str, int]] = set()
    py_basenames: set[str] = set()
    for name in matched:
        e = locmap["nodes"][name]
        b = os.path.basename(e["file"])
        py_basenames.add(b)
        want.add((b, int(e["line"])))

    check("gcc -g 로 .debug_line 섹션이 생성됨",
          "debug_dump" not in rd.stderr.lower() and bool(decoded.strip()),
          rd.stderr.strip()[:200])

    # readelf --debug-dump=decodedline 파서
    got_pairs: set[tuple[str, int]] = set()
    cur_file: str | None = None
    for ln in decoded.splitlines():
        s = ln.strip()
        if not s:
            continue
        mhdr = re.match(r"(\S.*?):$", s)
        if mhdr:
            cur_file = os.path.basename(mhdr.group(1))
            continue
        minline = re.match(r"(\S+\.\w+)\s+(\d+)\s+0x[0-9a-fA-F]+", s)
        if minline:
            got_pairs.add((os.path.basename(minline.group(1)), int(minline.group(2))))
            continue
        mrow = re.match(r"\S+\s+(\d+)\s+0x[0-9a-fA-F]+", s)
        if mrow and cur_file:
            got_pairs.add((cur_file, int(mrow.group(1))))

    got_py = sorted(p for p in got_pairs if p[0].endswith(".py"))
    result["readelf_py_pairs"] = got_py
    result["readelf_decodedline_excerpt"] = "\n".join(
        ln for ln in decoded.splitlines() if ".py" in ln
    )[:3000]

    check(f".debug_line 에 원본 .py 파일명이 등장 ({sorted(py_basenames)})",
          py_basenames.issubset({p[0] for p in got_py}),
          f"found={sorted({p[0] for p in got_py})}")

    hit = sorted(want & got_pairs)
    check(f"투영한 (py파일,라인) 조합이 DWARF 라인테이블에 그대로 존재  {hit or 'NONE'}",
          hit == sorted(want),
          f"want={sorted(want)}  got_py_pairs={got_py}")
    result["matched_pairs_in_dwarf"] = hit

    if matched and shutil.which("gdb"):
        name = matched[0]
        fn = code_track._fn_name(name)
        e = locmap["nodes"][name]
        pyb, pyl = os.path.basename(e["file"]), int(e["line"])
        g = _run(["gdb", "--batch", "-nx",
                  "-ex", f"break {pyb}:{pyl}",
                  "-ex", f"info line {pyb}:{pyl}",
                  binary])
        out_txt = g.stdout + g.stderr
        result["gdb_probe"] = out_txt.strip()[:800]
        check(f"gdb: `break {pyb}:{pyl}` 가 노드 함수 `{fn}` 에 바인딩됨",
              (f"in {fn}" in out_txt) or (f"<{fn}" in out_txt) or (fn in out_txt and pyb in out_txt),
              out_txt.strip()[:400])

    return result


# --------------------------------------------------------------------------- #
# 파이프라인
# --------------------------------------------------------------------------- #
def run(onnx_path: str, workdir: str, relative_to: str | None) -> dict:
    os.makedirs(workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]

    locmap_path = os.path.join(workdir, f"{stem}.locmap.json")
    c_path = os.path.join(workdir, f"{stem}_kernels.c")
    line_c_name = f"{stem}_kernels.line.c"
    line_c_path = os.path.join(workdir, line_c_name)
    bin_path = os.path.join(workdir, f"{stem}_kernels_dbg")

    # 1) 정보 트랙
    locmap = info_track.build_locmap(onnx_path, relative_to=relative_to)
    with open(locmap_path, "w", encoding="utf-8") as fh:
        fh.write(info_track.dumps_deterministic(locmap))

    # 2) 코드 트랙 (독립)
    c_src = code_track.get_generated_c(onnx_path)
    with open(c_path, "w", encoding="utf-8") as fh:
        fh.write(c_src)

    # 3) 조인 (유일한 만남)
    annotated, stats = join.inject_with_stats(
        c_src, locmap, out_c_name=line_c_name, resolve_paths=True
    )
    with open(line_c_path, "w", encoding="utf-8") as fh:
        fh.write(annotated)

    # 4) gcc -g
    comp = _run(["gcc", "-g", "-O0", "-o", bin_path, line_c_path])
    if comp.returncode != 0:
        return {"ok": False, "error": "gcc 실패", "stderr": comp.stderr}

    # 5) 검증
    verdict = verify_dwarf(bin_path, locmap, stats["matched"])

    report = {
        "inputs": {"onnx": os.path.abspath(onnx_path), "onnx_file": os.path.basename(onnx_path)},
        "generated": {
            "locmap": os.path.abspath(locmap_path),
            "kernels_c": os.path.abspath(c_path),
            "injected_c": os.path.abspath(line_c_path),
            "binary": os.path.abspath(bin_path),
        },
        "inject_stats": stats,
        "verify": verdict,
    }
    with open(os.path.join(workdir, f"{stem}_pipeline_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def reproducibility_check(onnx_path: str, relative_to: str | None, n: int = 3) -> dict:
    """locmap 과 투영 C 를 n 회 만들어 바이트 동일한지 확인 (컴파일 산출물은 대상 아님)."""
    locmaps: list[str] = []
    injected: list[str] = []
    for _ in range(n):
        lm = info_track.dumps_deterministic(
            info_track.build_locmap(onnx_path, relative_to=relative_to)
        )
        c = code_track.get_generated_c(onnx_path)
        inj = join.inject(c, json.loads(lm), out_c_name="x.line.c", resolve_paths=True)
        locmaps.append(lm)
        injected.append(inj)
    ok = len(set(locmaps)) == 1 and len(set(injected)) == 1
    return {"runs": n, "locmap_identical": len(set(locmaps)) == 1,
            "injected_c_identical": len(set(injected)) == 1, "ok": ok}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # 셸에서 stderr 와 순서 섞이지 않게
    ap = argparse.ArgumentParser(description="정보/코드 트랙 + 조인 오케스트레이터 (v3)")
    ap.add_argument("onnx", nargs="?",
                    default=os.path.join(HERE, "artifacts", "tinycnn_dynamo.onnx"),
                    help="dynamo=True .onnx (기본: artifacts/tinycnn_dynamo.onnx)")
    ap.add_argument("--workdir", default=os.path.join(HERE, "artifacts", "v3"),
                    help="산출물 디렉터리 (기본: artifacts/v3)")
    ap.add_argument("--relative-to", default=ROOT,
                    help="locmap file 경로 기준 디렉터리 (기본: 리포 루트)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.onnx):
        print(f"ONNX 없음: {args.onnx}\n"
              f"  python3 {os.path.join(ROOT, 'inspect_export_metadata.py')} 로 먼저 만드세요.",
              file=sys.stderr)
        return 2

    print("### 파이프라인: info_track → code_track → join → gcc -g → 검증 ###")
    report = run(args.onnx, args.workdir, args.relative_to)
    if not report.get("ok", True) and "verify" not in report:
        print(report.get("error", "실패") + "\n" + report.get("stderr", ""), file=sys.stderr)
        return 1

    verdict = report["verify"]
    print(json.dumps(verdict, ensure_ascii=False, indent=2))

    print("\n### 재현성 (info_track/code_track/join 3회 반복 바이트 비교) ###")
    rep = reproducibility_check(args.onnx, args.relative_to)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

    ok = verdict["ok"] and rep["ok"]
    print(f"\nV3 PIPELINE: {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    print(f"리포트: {os.path.join(args.workdir, os.path.splitext(os.path.basename(args.onnx))[0] + '_pipeline_report.json')}",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
