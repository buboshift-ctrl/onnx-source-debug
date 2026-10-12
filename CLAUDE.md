# 프로젝트: 재구성형 AI 프로세서 SW 프레임워크 - ONNX 디버깅 정보 보존 (4세부)

## 배경 (왜 이 작업을 하는가)
국가 R&D 과제 "재구성형 인공지능 프로세서 SW 프레임워크 기술개발"의 4세부(디버거·프로파일러·통합개발환경, 가천대 민홍 교수님 연구실 담당) 소속
학부생이 ONNX 파이프라인을 담당하게 되었다. 목표는 "ONNX은 원래 소스 위치 디버깅 정보를 남기지 않는다"는 문제를,
MLIR/DWARF 방식을 참고해서 실제로 디버깅(중단점, 변수 추적)이 가능하도록 설계하는 것.

## 실제 프로덕션 파이프라인 구조
```
ONNX 모델
  → (admit 단계 - 우리가 만짐)
  → [1-3] onnx → harx → EmitC   ← PNU가 만든 mlir-arx, LLVM 20, pinned 외부 컴파일러 (우리가 못 건드림, 라이선스 이슈로 더 조심)
  → [4-6] EmitC → C, requant repair, link  ← tail 단계, 우리가 소유
  → host-mode 실행 아티팩트
```
우리가 실제로 수정 가능한 지점은 **admit 단계(ONNX 로드 직후)**와 **tail 단계(EmitC→C 생성 이후)** 두 곳뿐이다.
중간 mlir-arx 컴파일러 내부는 절대 건드리지 않는다.

## 지금 다루는 이 도커 환경의 정체
- PNU의 ARX 포크가 아니라 **업스트림 onnx-mlir**(버전 0.5.1, LLVM 23, `src/Accelerators/`에 NNPA만 있고 ARX 없음).
- 그래서 실제 harx/EmitC/ARX 경로는 이 도커에서 재현 불가능하다.
- 대신 "같은 MLIR 인프라 위에서 provenance(위치 정보)가 얼마나 보존되는가"를 검증하는 **실험대**로 쓰고 있다.
- 이 도커는 네트워크가 되므로(claude code 설치가 curl로 성공했음) 필요하면 `pip install`도 가능하다. 다만 뭘 설치하기 전엔 항상 먼저 알리고 확인받을 것.

## 지금까지 확인/설계된 것 (다른 세션에서 이미 검증 완료)
1. `docs/mnist_example/mnist.onnx`를 `--EmitONNXIR --mlir-print-debuginfo`로 돌려서, ONNX의 `onnx_node_name` 속성과
   MLIR `loc("MaxPool_0")` 같은 Location 정보가 실제로 살아있는 것을 확인함. 단, 이 모델은 `doc_string`이 비어 있었음(구식 export 방식으로 만들어진 것으로 추정).
2. (별도 환경에서 검증) tail 단계 설계: 생성된 C 코드에 `#line` 지시자를 심으면 표준 `gcc -g` / DWARF / gdb가 그대로 동작한다는 것을
   `readelf --debug-dump=decodedline`으로 실증함. EmitC 경로는 MLIR→LLVM dialect를 거치지 않아서 정식 DWARF lowering이 없기 때문에 택한 우회 방법.
3. **PyTorch → ONNX export 자체의 현재 상태 조사 결과**(방금 확인):
   - 구식(TorchScript 기반) exporter는 `doc_string`에 스택 트레이스를 넣을 수 있었지만 기본값(`strip_doc_string=True` 계열)이 "제거"라서 사실상 안 남음.
   - 신식(`torch.onnx.export(..., dynamo=True)`, torch.export 기반) exporter는 **각 ONNX node에 자동으로 provenance 메타데이터를 붙인다**
     — 원본 PyTorch 모듈/메서드 계층(namespace 문자열), 원본 FX 노드의 문자열 표현 등. 이게 `doc_string`이 아니라 ONNX의
     `metadata_props`(node별 key-value 리스트) 필드에 들어간다.
   - **아직 확인 안 된 것**: 이 `metadata_props`를 onnx-mlir이 실제로 읽어서 `loc()`이나 다른 형태로 보존하는지 여부. 지금까지 확인한 건
     `onnx_node_name`(연산 이름, "Conv_0" 같은 것) 기반 보존뿐이고, `metadata_props`(PyTorch 원본 소스 계층 정보)는 별개로 테스트해야 함.

## 지금 요청하는 작업 (우선순위 순)

### 1순위 — PyTorch export 시점의 provenance 실제 내용 확인
`inspect_export_metadata.py`(별도 첨부, 이 폴더에 없으면 요청) 또는 그와 동등한 스크립트를 이 도커(또는 torch가 있는 다른 위치)에서 실행해서:
- `torch.onnx.export(..., dynamo=True)`로 만든 ONNX 파일의 각 node에 `metadata_props`가 실제로 어떤 내용을 담고 있는지 출력
- 구식(`dynamo=False`) exporter와 비교해서 `doc_string`이 정말 비어있는지 확인
- PyTorch가 이 컨테이너에 없으면 먼저 있는지 확인하고, 설치가 필요하면 설치 전에 반드시 물어볼 것 (`pip install torch onnx`)

### 2순위 — metadata_props가 onnx-mlir 임포터에서 살아남는지 확인
1순위에서 만든 `dynamo=True` 기반 ONNX 파일을 `build/Debug/bin/onnx-mlir --EmitONNXIR --mlir-print-debuginfo`로 돌려서
`metadata_props`에 있던 정보가 MLIR `loc()`이나 다른 attribute로 옮겨오는지, 아니면 통째로 버려지는지 확인.
- 살아남으면: admit 단계는 "새로 만들기"가 아니라 "이미 있는 걸 옮겨담기"로 설계를 단순화할 수 있음
- 버려지면: admit 단계에서 우리가 직접 원본 ONNX 파일의 `metadata_props`(또는 PyTorch FX 그래프의 `node.meta['stack_trace']`)를 별도로 읽어서
  캡처하는 우리만의 locmap 생성 로직이 필요해짐

### 3순위 — (이전에 요청했던 것, 우선순위 밀림) lowering 단계에서 provenance 생존 여부
`--EmitMLIR`, 가능하면 `--EmitLLVMIR`까지 낮추면서 loc 정보가 살아남는지, `FusedLoc`으로 합쳐지는지, 사라지는지 확인.
가능하면 `--EmitLib`로 실제 `.so`까지 만들어서 LLVM dialect가 정식 DWARF(`DISubprogramAttr` 등)를 만들 수 있는지도 확인 —
우리의 `#line` 우회 설계와 비교할 기준 마련.

### 공통: 결과 기록
매 단계 결과를 이 폴더 안에 markdown 문서(예: `debug_info_investigation.md`)로 누적 정리할 것. 나중에 부산대·경북대에 보낼
"Debugger 연계 협의" 자료(입력 대상 / 관측 지점 / 출력 형식)에 그대로 재사용할 예정이라, 각 실험의 결론을 짧게 요약해서 남겨야 함.

## 주의사항
- 명령 실행이나 파일 수정 전에는 항상 뭘 할 건지 먼저 설명하고 확인받을 것. `--dangerously-skip-permissions` 쓰지 말 것.
- `third_party/`, `build/`는 재빌드가 필요해지니 함부로 건드리지 말 것.
- 이 도커에는 PyTorch가 없을 수 있음 — 있는지부터 확인하고, 없으면 설치 여부를 먼저 물어볼 것.
- 이 컨테이너는 PNU의 ARX 컴파일러가 아니므로, "ARX/harx 경로를 재현했다"는 식의 결론은 내리지 말 것 — 항상 "일반 MLIR 인프라 기준"이라는 전제를 명시할 것.
"내가 돌아올 때까지 완료해줘"라고 요청을 보낼 것인데 이때는 별도의 허락받지 않고 네 양심에 따라 적절한 코드 수행할 것 위험한 파일은 절대 안됌. 사용자 요청완료시 수행한 결과 보고 파일 위치 알려주고 종료.

## 완료된 작업 기록 — v2 (2026-09-02)

전체 결과: `debug_info_investigation/debug_info_investigation.md` 2부. 재현: `debug_info_investigation/run_pipeline.sh`.

- **Part A (완료)** — `debug_info_investigation/locmap_capture.py`. `dynamo=True` ONNX 를 읽어
  `node.name`(=onnx-mlir `onnx_node_name`) → {file,line,source,module_scope,class_hierarchy,fx_node,stack_frames}
  locmap JSON 을 만든다. `tinycnn_dynamo.onnx` 5/5 노드 매핑, onnx_node_name 집합과 키 집합 일치, 3회 반복 바이트 동일.
- **Part A2 (실패로 기록, 타임박스 준수)** — `--enable-debug-info` 는 stock `DIScopeForLLVMFuncOpPass` 가
  onnx-mlir 의 모든 생성 함수에 `DISubprogram` 을 붙이는데 그 안 런타임 헬퍼 호출들이 `loc(unknown)` 이라
  MLIR LLVM verifier 에 걸려 실패. 고치려면 새 패스+전체 재빌드 필요 → 소스 미수정, 진단만 기록
  (`artifacts/partA2_enable_debug_info_diagnosis.txt`). 참고: 통과시켜도 모델 loc 이 NameLoc 이라 line 0 이 됨.
- **Part B (완료)** — `debug_info_investigation/inject_line_directives.py`. 생성 C 의 `onnx_node_name` 마커에
  locmap 을 조인해 `#line <원본.py>` 를 투영. `--demo` 로 EmitC 풍 C → `gcc -g` → `readelf`/`gdb` 검증 4/4 PASS:
  원본 `.py:21..24` 가 DWARF 라인테이블에 그대로, `gdb break inspect_export_metadata.py:21` 이 해당 노드
  컴파일 코드(`node_node_max_pool2d+16`)에 바인딩됨. 3회 반복 바이트 동일.
- **RTST 매핑** — "AI 포맷 로드"/"레이어별 정보 측정"(+locmap 스키마 표)/"DNN 연산자 지원"/"반복 재현성" 근거를 2부에 기록.
- 이 컨테이너엔 실제 EmitC 경로가 없어 Part B 는 데모 C 로 실증함. 실제 harx/EmitC 출력에 `onnx_node_name` 이
  남는지 확인 + admit/tail 파이프라인 결선은 후속 과제(2부 "남은 일" 참고).

## 알려진 범위 제한
locmap 캡처(stack_trace 기반)는 PyTorch(torch.onnx.export dynamo=True)에서만 검증됨.
TensorFlow/scikit-learn 등 다른 프레임워크의 ONNX 변환기가 동등한 metadata_props를 채우는지는 미확인.
2026년 검증 벤치마크(ResNet-18, MobileNetV2)가 전부 TorchVision 기반이라 당장은 문제 없지만,
다른 프레임워크 모델이 들어오면 이 가정을 재검증해야 함.

코드 트랙(code_track.py)은 현재 데모 C 생성기로 대체돼 있음 — 실제 harx/EmitC 출력이 아님.
부산대로부터 실제 EmitC 샘플을 받으면 code_track.get_generated_c()의 구현부만 교체하면 되도록 설계함.

## 완료된 작업 기록 — v3 구조 정리 (2026-09-03, TASK_formalize_architecture_v3.md)

v2에서 검증된 두 트랙 구조를 순수 리팩터링으로 모듈화. 새 기술 검증 없음, 회귀 없음.
상세: `debug_info_investigation/debug_info_investigation.md` 3부.

- **모듈 분리** (`debug_info_investigation/`):
  - `info_track.py` — 정보 트랙. `build_locmap(onnx_path) -> dict`. 코드 트랙과 완전 독립.
  - `code_track.py` — 코드 트랙 대역. `get_generated_c(onnx_path) -> str`. ONNX 그래프에서
    `node.name`/`op_type`만 읽음(provenance 안 읽음). **실제 harx/EmitC 연동 시 이 함수 구현부만 교체**
    (시그니처: 경로 → C 문자열 유지). docstring에 명시.
  - `join.py` — 유일한 만남 지점. `inject(c_source, locmap) -> str`. 조인 키 = `node.name`.
  - `run_full_pipeline.py` — 오케스트레이터(build_locmap → get_generated_c → inject → gcc -g →
    readelf/gdb 검증 → 3회 재현성).
  - `locmap_capture.py` / `inject_line_directives.py` — 하위호환 shim으로 축소(기존 명령 그대로 동작).
  - `run_pipeline.sh` — `run_full_pipeline.py`를 부르는 얇은 셸.
- **회귀 확인 (v2와 동일)**: locmap 바이트 동일, `readelf` `.py:21..24` 동일,
  `gdb break inspect_export_metadata.py:21` → `0x1159 <node_node_max_pool2d+16>` 문자열까지 동일,
  4/4 검증 통과, 3회 반복 바이트 동일. 산출물: `debug_info_investigation/artifacts/v3/`.
