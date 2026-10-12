# Lab 기록 — `#line` 기반 ONNX node → 원본 소스 매핑 파이프라인 실행 검증

- 작성일: 2026-09-11
- 대상 코드: `debug_info_investigation/{info_track.py, code_track.py, join.py, run_full_pipeline.py}`
- 목적: "ONNX admit 단계에서 뽑은 위치 정보를, tail 단계 생성 C에 `onnx_node_name` 기준
  `#line` 지시자로 다시 투영하는" 로직이 실제로 동작하는지 재실행으로 확인.
- 결론: **이미 구현·검증 완료된 상태**임을 재확인 (PASS). 새 코드 작성 없음 — 순수 실행 로그.

## 0. 배경 (파이프라인 구조 요약)

```
ONNX(metadata_props 보존)
  ├─ info_track.py  (admit)  : node.metadata_props["pkg.torch.onnx.stack_trace"] 파싱
  │                             → locmap = { node.name : {file, line, source, ...} }
  └─ code_track.py  (tail 대역): 이 컨테이너엔 실제 harx/EmitC가 없어 데모 C 생성기로 대체.
                                  "/* onnx_node_name: <name> */" 마커가 붙은 C를 만듦
         │
         └─ join.py             : 마커로 두 트랙을 node.name 기준 조인, 그 자리에
                                   `#line <원본줄> "<원본.py>"` 삽입 → gcc -g → DWARF에 원본 .py 좌표 보존
```

## 1. 실행 방법

### 1-1. 전제 조건
- 이 컨테이너의 시스템 `python3`(`/usr/bin/python3`)에는 `onnx`/`torch`가 없음.
- `onnx`/`torch`가 설치된 venv가 `/home/ljy/venv`에 이미 존재함 (`onnx 1.22.0`, `torch 2.14.0+cpu`).
  → 반드시 이 venv의 python으로 실행해야 함.
- `debug_info_investigation/artifacts/` 아래 기존 산출물은 이전 세션에서 **root 권한**으로 생성되어
  현재 사용자(`ljy`)로는 덮어쓸 수 없음(`PermissionError`). 따라서 `--workdir`을 별도 쓰기 가능한
  경로로 지정해서 실행함(아래 예시는 스크래치 경로 사용; 필요하면 `~/onnx-mlir/lab` 하위로 지정해도 됨).

### 1-2. 실행 명령

```bash
cd ~/onnx-mlir/debug_info_investigation

# 기본 입력(artifacts/tinycnn_dynamo.onnx)으로, 결과를 쓰기 가능한 디렉터리에 저장
/home/ljy/venv/bin/python3 run_full_pipeline.py \
  --workdir /path/to/writable/dir
```

옵션:
- 인자 없이 첫 번째 위치 인자를 주면 다른 `.onnx`(dynamo=True export)로도 실행 가능:
  `run_full_pipeline.py OTHER_MODEL.onnx --workdir DIR`
- `--relative-to DIR` : locmap에 기록되는 원본 `.py` 경로를 이 디렉터리 기준 상대경로로 변환(기본: repo 루트)

### 1-3. 내부적으로 수행되는 단계
1. `info_track.build_locmap(onnx_path)` — ONNX 로드 후 노드별 locmap dict 생성, JSON으로 저장
2. `code_track.get_generated_c(onnx_path)` — 노드당 함수 1개, `onnx_node_name` 마커가 붙은 데모 C 생성
3. `join.inject_with_stats(c_src, locmap, ...)` — 마커 뒤에 `#line` 삽입, 노드 블록 끝나면 원복
4. `gcc -g -O0` 로 컴파일
5. `readelf --debug-dump=decodedline` 으로 DWARF 라인테이블에 원본 `.py:line`이 실제로 박혔는지 확인
6. `gdb --batch` 로 `break <원본.py>:<line>` 이 해당 노드 함수 주소에 바인딩되는지 확인
7. 위 1~3을 3회 반복해 locmap/투영된 C가 바이트 단위로 동일한지(재현성) 확인

## 2. 실행 결과 (2026-09-11, 이 세션에서 재실행)

입력: `debug_info_investigation/artifacts/tinycnn_dynamo.onnx` (5-노드 TinyCNN, dynamo=True export)

### 2-1. 조인 통계 (`inject_stats`)
```json
{
  "matched": ["node_max_pool2d", "node_view", "node_linear", "node_relu", "node_linear_1"],
  "unknown": [],
  "directives": 10,
  "nodes_in_locmap": 5
}
```
→ locmap의 노드 5개가 전부 C 마커와 매칭됨(미매칭 0). 매칭 노드당 `#line` 2개(진입+복귀) = 10개.

### 2-2. DWARF/gdb 검증 (`verify.checks`) — 4/4 전부 `ok: true`

| # | 검사 항목 | 결과 |
|---|---|---|
| 1 | `gcc -g` 로 `.debug_line` 섹션 생성됨 | PASS |
| 2 | `.debug_line`에 원본 파일명(`inspect_export_metadata.py`) 등장 | PASS |
| 3 | 투영한 (py파일,라인) 4쌍이 DWARF 라인테이블에 그대로 존재<br>`[('inspect_export_metadata.py',21..24)]` | PASS |
| 4 | `gdb break inspect_export_metadata.py:21` 이 노드 함수 `node_node_max_pool2d`에 바인딩 | PASS |

gdb 실측 출력:
```
Breakpoint 1 at 0x1159: file /workdir/onnx-mlir/inspect_export_metadata.py, line 21.
Line 21 of "/workdir/onnx-mlir/inspect_export_metadata.py" starts at address
0x1159 <node_node_max_pool2d+16> and ends at 0x1161 <node_node_max_pool2d+24>.
```
→ 원본 PyTorch 소스(`inspect_export_metadata.py`)의 21번째 줄에 건 중단점이, 생성 C의
`node_node_max_pool2d` 함수 코드 주소에 실제로 바인딩됨. `#line` 투영 → DWARF → gdb 경로가
end-to-end로 동작함을 실측 확인.

### 2-3. 재현성 (3회 반복)
```json
{ "runs": 3, "locmap_identical": true, "injected_c_identical": true, "ok": true }
```

### 2-4. 최종 판정
```
V3 PIPELINE: PASS
```

### 2-5. 산출물 (이번 실행분)
스크래치 디렉터리에 생성(참고용, 재실행 시마다 새로 생성됨):
- `tinycnn_dynamo.locmap.json` — info_track 산출 locmap
- `tinycnn_dynamo_kernels.c` — code_track 산출 데모 C (마커만 있고 `#line` 없음)
- `tinycnn_dynamo_kernels.line.c` — join 산출, `#line` 투영된 C
- `tinycnn_dynamo_kernels_dbg` — `gcc -g` 컴파일된 바이너리
- `tinycnn_dynamo_pipeline_report.json` — 위 표의 원본 JSON 리포트

기존 정본 산출물(참고): `debug_info_investigation/artifacts/v3/` (2026-09-03 v3 정리 시점,
동일 결과, root 소유라 현재 사용자로는 재생성 불가 — 필요시 쓰기 권한 있는 경로로 `--workdir` 지정해서 재실행).

## 4. 복잡한 모델 확장 검증 + 버그 발견/수정 + PPT 2종 (2026-09-14)

- **입력**: `lab/export_complex_models.py`로 만든 torchvision `resnet18`(49노드), `mobilenet_v2`(100노드) —
  둘 다 `dynamo=True` export, provenance(`pkg.torch.onnx.stack_trace`) 100% 매핑.
- **1차 실행 결과** (`lab/artifacts/resnet18/`, `lab/artifacts/mobilenet_v2/`): 파이프라인 4/4
  PASS·재현성 PASS 자체는 문제 없었으나, locmap의 (file,line)이 노드마다 전부 동일값으로
  수렴(resnet18: 49개 노드 → 서로 다른 줄 1개, mobilenet_v2: 100개 → 1개)하는 현상 발견.
- **원인 규명**: `info_track.py`의 `_pick_user_frame()`이 파이썬 stack trace의 "가장 바깥쪽"
  프레임을 선택하는데, `ResNet.forward → _forward_impl → BasicBlock.forward`처럼 forward가
  여러 겹인 모델에서는 가장 바깥쪽 프레임이 모든 연산에 대해 항상 동일하기 때문(TinyCNN은
  forward가 한 겹이라 우연히 문제가 안 드러났음).
- **수정**: `_pick_user_frame()`을 "가장 안쪽(마지막) 비-프레임워크 프레임" 우선으로 변경
  (`info_track.py` 함수 본문 교체, 문서 주석 포함). 재실행 결과(`lab/artifacts/resnet18_fixed/`,
  `lab/artifacts/mobilenet_v2_fixed/`): 서로 다른 줄이 resnet18 1→7개, mobilenet_v2 1→5개로
  개선, DWARF/gdb 검증 4/4 PASS·3회 재현성 PASS 유지(회귀 없음). 남은 한계: 같은 helper
  클래스를 여러 인스턴스가 재사용하면 "연산 종류" 단위까지만 구분됨 — 레이어 인스턴스 단위
  구분은 locmap의 `module_scope`를 추가로 써야 함.
- **별개 발견(uncommitted)**: `src/Builder/FrontendDialectTransformer.cpp`의 `ImportLoc()`/
  `ParseStackTrace()`에 이미 손으로 짜둔(커밋 안 됨) 패치가 있어, onnx-mlir 임포터 자체가
  `pkg.torch.onnx.stack_trace`를 `FileLineColLoc`으로 복원해 `NameLoc`과 `FusedLoc`으로
  합치고 있음을 확인. 동일한 "가장 바깥쪽 프레임 편향" 문제가 이쪽 C++ 파서에도 있음(향후 보완
  필요). 이 발견은 논문 초안 자료로 별도 정리(`lab/paper_draft.pptx`), 과제 파이프라인
  (`info_track/code_track/join`)과는 독립적인 별도 실험이며 실제 ARX 파이프라인에는 적용 불가
  (mlir-arx 내부를 직접 고쳐야 하므로).
- **산출물**: `lab/report_assignment.pptx`(과제 보고, 방법1), `lab/paper_draft.pptx`(논문 초안,
  방법2), `lab/build_slides.py`(생성 스크립트, 재실행하면 최신 리포트 JSON으로 재생성됨).
- **손대지 않은 것**: 저장소 곳곳의 기존 커밋 안 된 파일들(`docs/docker-example/Dockerfile`,
  `paper/m.onnx`, `paper/test_model.py`, `src/testcase.py`, `out.onnx.mlir`, `out.tmp`) —
  이번 작업과 무관해 보여 사용자 확인 없이 건드리지 않음.
- **산출물 인덱스**: `lab/artifacts/INDEX.md`에 위 두 실행(수정 전/후) 전체의 단계별 파일
  경로를 표로 정리해뒀다. 개별 JSON/C/MLIR 파일을 하나씩 찾을 필요 없이 이 표만 보면 됨.
- **방법2 수치 기록**: 대화 중 터미널 출력으로만 보여줬던 loc(...) 등장 횟수, FileLineColLoc
  개수, TinyCNN/ResNet-18/MobileNetV2 3개 모델 비교값을 `lab/artifacts/method2_importer_patch_findings.md`에
  파일로 남겼다(재현 명령 포함).

## 4-1. PPT에 코드+실제 실행 스크린샷 추가 (2026-09-14, 같은 세션 이어서)

처음 만든 두 PPT가 표/불릿 요약 위주였는데, 사용자가 "각 단계의 코드를 보여주면서 실행 결과
스크린샷도 포함해달라"고 정정했다. 이 컨테이너엔 GUI/브라우저가 없어 실제 화면 캡처는 불가능해서,
**실제로 실행해서 얻은 진짜 출력 텍스트**(가상 데이터 아님)를 터미널 창처럼 보이는 PNG로
렌더링하는 방식으로 대체했다.

- `lab/make_screenshots.py` — Pillow(`ImageFont.load_default(size=N)`, 시스템에 한글 폰트가
  없어 이미지 안 텍스트는 전부 영어로 작성, 한글 설명은 슬라이드 쪽 네이티브 텍스트로 분리)로
  8장의 "터미널 스크린샷"을 `lab/shots/`에 생성. 각 이미지의 실제 데이터 출처(파일 경로/명령)는
  스크립트 주석에 남겨둠 — resnet18_fixed의 `node_Conv_292` 노드를 공통 예시로 관통시켜서
  info_track→code_track→join→gcc→verify 5단계가 같은 노드를 따라가는 걸 보여줌.
- `build_slides.py`에 `code_shot_slide()`/`bullet_shot_slide()`(좌: 코드 또는 설명 텍스트,
  우: 실제 출력 스크린샷, 이미지는 비율 유지 + 세로 중앙 정렬 + 캡션이 이미지 바로 아래 따라붙음)
  추가. 과제 PPT는 10→15슬라이드(단계①~⑤ 5장 추가), 논문 초안 PPT는 슬라이드 수 그대로 10장
  유지(기존 code_slide/bullet_slide 4장을 스크린샷 포함 버전으로 교체).
- 재생성: `python3 make_screenshots.py && python3 build_slides.py` (둘 다 `lab/` 안에서 실행).

## 4-2. v3 폴더 정리 + 세 모델 통합 실행 스크립트 (2026-09-14, 같은 세션 이어서)

"v3 폴더가 지저분하다"는 지적을 받고 구조를 정리했다.

- **이전 상태**: `debug_info_investigation/artifacts/v3/`엔 tinycnn 결과 파일 6개가 폴더 없이
  평평하게 있었고, resnet18/mobilenet_v2의 "수정 후(최신)" 결과는 `lab/artifacts/resnet18_fixed/`,
  `lab/artifacts/mobilenet_v2_fixed/`에 따로 있어서 위치 규칙이 모델마다 달랐다.
- **정리 후**: 세 모델 전부 `debug_info_investigation/artifacts/v3/<모델명>/` 밑에 같은 파일
  구성(`*.locmap.json`, `*_kernels.c`, `*_kernels.line.c`, `*_kernels_dbg`,
  `*_pipeline_report.json`)으로 통일. `lab/artifacts/resnet18_fixed/`,
  `mobilenet_v2_fixed/`는 이동 후 삭제(빈 폴더만 rmdir). **"버그 수정 전(1차 실행)" 결과**
  (`lab/artifacts/resnet18/`, `mobilenet_v2/`)는 비교 근거로 의도적으로 그대로 남겨둠 — 옮기지 않음.
- `debug_info_investigation/run_full_pipeline.py`의 `--workdir` 기본값을 고정 경로(`artifacts/v3`)에서
  **모델명 자동 분리**(`artifacts/v3/<onnx파일명, 확장자 제외>/`)로 변경. `run_pipeline.sh`도
  동일 규칙으로 맞춤 — 앞으로 `--workdir` 안 주고 그냥 돌리면 알아서 모델별 폴더가 생긴다.
- `lab/run_examples.sh`를 resnet18/mobilenet_v2 2개 전용에서 **tinycnn_dynamo까지 포함한
  3개 모델 통합 스크립트**로 확장. 산출물 위치도 `lab/artifacts/<model>_fixed/`에서
  `debug_info_investigation/artifacts/v3/<model>/`로 통일.
- 재실행해서 3개 모델 전부 회귀 없음 확인: `tinycnn_dynamo/resnet18_dynamo/mobilenet_v2_dynamo`
  전부 `verify.ok=true`, matched=5/49/100 unknown=0, 4개 체크 전부 PASS.
- `build_slides.py`의 파일 경로도 새 위치로 갱신하고 두 PPT 재생성(내용 수치는 동일, 파일
  위치만 바뀐 것이므로 슬라이드 텍스트 변경 없음).
- **경로가 바뀐 파일을 참조하던 문서**: `lab/artifacts/INDEX.md` 전면 재작성, 이 문서(4-2절 신설).
  `4절`/`4-1절` 본문에 남아있는 `_fixed` 경로 언급은 "그 시점 기준" 기록이라 그대로 두되, 최신
  위치는 이 절과 INDEX.md를 보면 된다.
- **부수 발견**: 이 작업 도중 `torch`/`torchvision`/`python-pptx`가 같은 세션 안에서 갑자기
  사라지는 걸 겪었다 — 컨테이너가 대화 중간에 바뀐 것으로 확인됨. 자세한 내용과 대응은
  `ENVIRONMENT.md`의 "같은 대화 도중에도 컨테이너가 바뀔 수 있다" 절 참고.

## 5. 환경은 세션마다 달라짐 — 별도 스냅샷 문서로 분리 (2026-09-14)

이 세션 도중 사용자가 "1절에 '실행 불가/venv 필요'라고 적혀 있는데 어떻게 실행했냐"고 질문해서
확인해보니, **1절(2026-09-11)이 전제한 컨테이너 상태(`ljy` 사용자, `/home/ljy/venv`)가 이번
세션의 컨테이너에는 아예 존재하지 않았다** — 이번 세션은 `root`였고 시스템 `python3`에
`torch`/`onnx`가 이미 전역 설치되어 있었다. 즉 이 문서(`lab.md`)에 적힌 "실행 방법"은 그걸 쓴
세션의 컨테이너에만 유효할 수 있고, 다음 세션에 그대로 안 맞을 수 있다.

그래서 "환경이 이랬다"는 서술은 `lab.md` 본문에 계속 누적하지 않고, 매 세션 갱신 가능한 별도
문서 **`lab/ENVIRONMENT.md`**로 분리했다. 다음 세션은 이 문서의 "확인 절차"를 먼저 실행해서
새 스냅샷을 그 파일에 추가할 것 — `lab.md`나 과거 스냅샷을 실행 방법의 근거로 그대로 믿지 말 것.

## 3. 남은 범위 제한 (재확인, 이번 실행으로 바뀐 것 없음)

`debug_info_investigation/debug_info_investigation.md` 399줄 "남은 일" 절 그대로:
1. `code_track.py`는 데모 C 생성기 — 실제 PNU harx/EmitC 산출물로 검증 안 됨(이 컨테이너에 없음).
2. admit 파이프라인에 `info_track`을 실제 엔드포인트로 붙이는 배선은 이 저장소 범위 밖.
3. 노드=함수 1:1 가정의 `#line` 세분도가 실제 EmitC 출력에도 적절한지는 디버거팀과 별도 협의 필요.
4. MLIR 단계 `FileLineColLoc` 주입은 Part A2에서 실패로 진단·기록됨(우선순위 낮음, 보류).
