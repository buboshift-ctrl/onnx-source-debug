# 작업 지시: ONNX 디버깅 정보 캡처 → DWARF 투영 구현 (v2)

- 전제 문서(반드시 먼저 참고): `CLAUDE.md`, `debug_info_investigation.md`, `TASK_implement_dwarf_pipeline.md`(v1, 이 문서로 대체됨 — 삭제하지 말고 참고용으로 남겨둘 것)
- v1 대비 바뀐 점: **"진짜 MLIR→LLVM DWARF 경로(`--enable-debug-info`) 복구 시도"를 타임박스 걸고 병행 추가**, RTST 검증 요청 항목 중 ONNX 관련 부분을 명시적으로 매핑.

## 확정된 전제 (v1과 동일, 재논쟁 금지)

1. onnx-mlir 임포터는 `metadata_props`/`doc_string`을 버리고 `node.name()`만 보존한다. **onnx-mlir/third_party는 수정하지 않는다** (patch 시도해도 되는 유일한 예외는 아래 A2 항목뿐이며, 그것도 이 컨테이너 안에서의 실험용이지 실제 ARX 파이프라인에 반영하려는 게 아니다).
2. provenance 원천은 `torch.onnx.export(..., dynamo=True)`의 node `metadata_props`, 특히 `pkg.torch.onnx.stack_trace`.
3. 조인 키는 `node.name` = onnx-mlir의 `onnx_node_name`.
4. **왜 `--enable-debug-info`를 고치는 게 우선순위가 아닌지**: 이 경로는 onnx-mlir 자체의 LLVM dialect 백엔드(`--EmitObj`/`--EmitLib`)에만 관련 있다. 실제 프로덕션(ARX) 파이프라인은 `EmitC → 외부 C 컴파일러`로 빠져서 LLVM dialect를 아예 안 거치므로, 이 경로를 고쳐도 ARX 파이프라인에는 적용되지 않는다. **그래도 완전히 무의미하진 않다** — onnx-mlir 자체 CPU 실행 경로가 검증 시나리오의 "레퍼런스 실행"으로 쓰일 수 있어서, 되면 좋고 안 되면 접는 병행 작업으로 다룬다.

## 작업 순서 (v1에서 재배치)

### Part A — admit 단계: locmap 캡처 도구 (그대로, 1순위, 필수)
v1과 동일. `debug_info_investigation/locmap_capture.py`. 이 항목은 A2 결과와 무관하게 반드시 완료한다.

### Part A2 — (신규, 타임박스, 선택적) 진짜 MLIR→LLVM DWARF 경로 복구 시도

**목표**: `--enable-debug-info` 사용 시 나는 verifier 에러(`inlinable function call ... must have a debug location`)의 정확한 발생 지점을 찾아서, 고칠 수 있으면 고친다.

**절차**:
1. `--mlir-print-ir-after-failure`로 정확히 어느 pass·어느 op에서 에러가 나는지 확인.
2. 원인이 "런타임 헬퍼 호출부(`omTensor*` 등)에 debug location이 없다"는 게 맞는지 확인 — 맞다면, `DIScopeForLLVMFuncOpPass` 실행 전에 그 op들에 fallback debug location(예: 함수 전체를 감싸는 `DISubprogram` 위치로 통일)을 채워주는 **작은 프리패스**로 해결되는지 시도.
3. `src/Conversion/KrnlToLLVM/KrnlInstrument.cpp`가 `DICompileUnitAttr`/`DISubprogramAttr`를 수동으로 만드는 기존 코드가 있으니, 그 패턴을 참고해서 최소 변경으로 시도.

**타임박스 (반드시 지킬 것)**:
- **이 항목에 반나절 이상 쓰지 않는다.** 원인 파악에 1~2시간 넘게 걸리거나, 고치려면 여러 pass를 건드려야 하는 게 명확해지면 **그 시점에서 바로 중단**하고 `Part A2는 실패로 기록, Part B로 진행`.
- 성공하든 실패하든 결과(무엇을 시도했고 왜 안 됐는지/됐는지)는 반드시 `debug_info_investigation.md`에 기록. "시간 없어서 안 함"이 아니라 "이런 원인 때문에 이 시간 안엔 안 풀림"이 남아야 함.

### Part B — tail 단계: `#line` 투영 도구 (그대로, 반드시 완료, A2와 무관하게 진행)
v1과 동일. `debug_info_investigation/inject_line_directives.py`. A2가 진행 중이든 끝났든 이 작업은 병행/계속 진행한다 — ARX 파이프라인에는 결국 이 경로가 필요하기 때문.

## RTST 검증 요청 항목 매핑 (덤 요청 — 문서화만 하면 됨, 별도 대규모 구현 아님)

디버거 검증 요청 7개 중 이 작업과 관련된 것만:

| 요청 항목 | 이 작업과의 관계 | 해야 할 일 |
|---|---|---|
| AI 포맷 로드 | 우리 도구(`locmap_capture.py`)가 ONNX(dynamo export)를 실제로 로드해서 디버깅 정보를 추출하는지 여부 | Part A 완료 = 이 항목 충족. 별도 구현 없음, 결과 문서에 "충족" 근거로 기록 |
| 레이어별 정보 측정 | "어떤 정보를 확인할지"의 초안이 곧 locmap 스키마(file/line/source/module_scope/fx_node) | Part A의 locmap 스키마를 이 항목의 제안 스펙으로 별도 문서화(간단한 표 하나면 됨) |
| DNN 연산자 지원 | locmap 설계가 `node.name` 기반이라 연산자 종류 무관하게 자동 커버됨 | 별도 구현 불필요. "설계상 이미 만족 — 연산자별 분기 없음"이라고 한 줄 기록만 |
| 동일 입력 일관된 출력 / 반복 실행 재현성 | DWARF 자체보다, 우리 스크립트(locmap 생성/`#line` 삽입)가 매번 같은 입력에 같은 출력을 내는지의 품질 기준 | 같은 `.onnx` 파일로 `locmap_capture.py`/`inject_line_directives.py`를 2~3회 반복 실행해서 출력이 바이트 단위로 동일한지 확인, 결과 기록 |

관련 없음(기록만 하고 손대지 않음): SNN 레이어별 발화율 측정, SNN 타임스텝 지원 — 둘 다 SNN 담당(UNIST 이종은 교수님 연구실) 몫이라 이 작업 범위 밖.

## Definition of Done (갱신)

- [ ] Part A: `locmap_capture.py` 완료, `tinycnn_dynamo.onnx`로 검증
- [ ] Part A2: 성공/실패 여부와 원인이 기록됨 (타임박스 준수 여부 포함)
- [ ] Part B: `inject_line_directives.py` 완료, 진짜 `.py` 파일:라인이 DWARF에 찍히는 것을 readelf로 확인
- [ ] 재현성 체크: A/B 스크립트를 2~3회 반복 실행해 출력 동일함을 확인
- [ ] RTST 매핑 표의 "AI 포맷 로드"/"레이어별 정보 측정" 항목에 대해 결과 문서에 한 문단씩 근거 기록
- [ ] `debug_info_investigation.md`에 전체 결과가 이어붙여짐 (기존 내용 삭제 금지)

## 하지 말 것

- onnx-mlir/`third_party` 수정은 A2 실험 목적 외에는 금지, A2도 타임박스 넘기면 원복하고 중단
- 새 패키지 설치 전 확인받기
- Part A2가 안 풀렸는데 성공한 것처럼 기록하지 말 것 — 실패도 유효한 결과
- SNN 관련 항목(레이어별 발화율, 타임스텝)에는 시간 쓰지 말 것 — 범위 밖
