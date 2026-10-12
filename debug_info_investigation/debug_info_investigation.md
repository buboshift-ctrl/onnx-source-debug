# ONNX 디버깅 정보 보존 조사 (debug_info_investigation)

- 작성일: 2026-09-02
- 환경: 업스트림 onnx-mlir `v.0.5.1.0-69-g706eb3d6`, LLVM 23, `build/Debug/bin/onnx-mlir`
- 추가 설치(네트워크 사용): `torch 2.14.0+cpu`, `onnxscript 0.7.1`, `onnx_ir 1.0.0` (`onnx 1.21.0`은 사전 설치됨)
  - 설치는 `pip install --break-system-packages` 로 진행. `torch`는 CPU 전용 휠(196MB).
- 재현 스크립트: `debug_info_investigation/run_investigation.sh`
- 산출물: `debug_info_investigation/artifacts/` (ONNX 파일, 각 단계 `.onnx.mlir`, 로그)
- **전제**: 이 컨테이너는 PNU의 ARX/harx 컴파일러가 아니다. 아래 결론은 전부 **"일반 MLIR/onnx-mlir 인프라 기준"**이며 ARX 경로 재현이 아니다.

---

## 요약 (한 줄 결론)

| 항목 | 결론 |
|---|---|
| PyTorch `dynamo=True` export 시 provenance | **node별 `metadata_props`에 원본 소스 파일·라인·스택트레이스·모듈 계층이 전부 들어간다** |
| PyTorch 구식(`dynamo=False`) export | torch 2.14에서는 node `doc_string`·`metadata_props` **모두 빈 값**. (구버전 torch 1.12로 만든 리포의 `mnist.onnx`는 node `doc_string`에 스택트레이스가 있음 — 버전 의존적) |
| onnx-mlir 임포터가 `metadata_props` / `doc_string`을 읽는가 | **아니오. 완전히 버린다.** 소스에 `metadata_props`·`doc_string` 참조가 0건. `node.name()`만 사용 |
| onnx-mlir 임포터가 보존하는 것 | `node.name()` → `loc("<name>")` (NameLoc) + `onnx_node_name` 속성. 그게 전부 |
| `loc()`가 lowering에서 살아남는가 | **살아남는다.** `--EmitONNXIR` → `--EmitMLIR`(krnl/affine) → `--EmitLLVMIR`(LLVM dialect) 까지 모든 파생 op에 전파됨. `NameLoc` 중첩(`loc("onnx.Gemm"(loc("node_linear")))`), `FusedLoc` 아님, 소실 아님 |
| 정식 DWARF 생성(`--enable-debug-info`) | **깨져 있다.** `DIScopeForLLVMFuncOpPass`가 MLIR LLVM verifier 에러(`inlinable function call ... must have a debug location`)로 실패. mnist·dynamo 모델 모두 |
| 완성된 `.so`의 DWARF | 있긴 하지만 **onnx-mlir C 런타임(`OMTensor.c` 등)에서 온 것뿐**. 모델 연산 코드(`main_graph`)에는 라인 정보 없음 |

**admit 단계 설계 함의**: onnx-mlir이 provenance를 버리므로, admit 단계는 "이미 있는 걸 옮겨담기"가 아니라 **우리만의 locmap 캡처 로직이 필요**하다 (원본 ONNX의 `node.metadata_props` 또는 FX 그래프의 `node.meta['stack_trace']`를 직접 읽어서 `onnx_node_name` ↔ 소스위치 매핑을 별도 파일로).

**tail 단계 설계 함의**: 정식 DWARF 경로가 동작하지 않으므로, 생성된 C 코드에 `#line` 지시자를 심는 우회 설계가 (현 시점 업스트림 기준) 유일하게 현실적인 선택지라는 점이 재확인됨.

---

## 1순위 — PyTorch export 시점의 provenance 실제 내용

### 실험

`inspect_export_metadata.py`(리포 첨부)를 실행. `TinyCNN`(MaxPool → flatten → Linear → ReLU → Linear)을 두 방식으로 export:

- `torch.onnx.export(..., dynamo=True)` → `tinycnn_dynamo.onnx` (opset 20, IR v10, `torch.export` 기반)
- `torch.onnx.export(..., dynamo=False)` → `tinycnn_legacy.onnx` (TorchScript 기반, "legacy" deprecation 경고 출력)

### 결과 A: `dynamo=True` — node마다 `metadata_props` 5종

예 (`MaxPool` node, `node_max_pool2d`):

```
metadata_props[namespace]                    = : __main__.TinyCNN/pool: torch.nn.modules.pooling.MaxPool2d/max_pool2d: aten.max_pool2d.default
metadata_props[pkg.torch.onnx.class_hierarchy] = ['__main__.TinyCNN', 'torch.nn.modules.pooling.MaxPool2d', 'aten.max_pool2d.default']
metadata_props[pkg.torch.onnx.fx_node]       = %max_pool2d : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d.default](args = (%x, [2, 2], [2, 2]), kwargs = {})
metadata_props[pkg.torch.onnx.name_scopes]   = ['', 'pool', 'max_pool2d']
metadata_props[pkg.torch.onnx.stack_trace]   = File ".../inspect_export_metadata.py", line 21, in forward
    x = self.pool(x)
  File ".../torch/nn/modules/pooling.py", line 224, in forward
    return F.max_pool2d( ...
```

- `doc_string` 은 `''` (빈 값) — provenance는 전부 `metadata_props` 로 이동.
- **`pkg.torch.onnx.stack_trace` 가 핵심**: 원본 `.py` 파일 경로 + 라인 번호 + 소스 코드 한 줄까지 그대로 들어있다. 중단점·변수 추적에 바로 쓸 수 있는 형태.
- `namespace` / `class_hierarchy` / `name_scopes` 는 모듈 계층(예: `TinyCNN.pool`)을 문자열로 제공 — 디버거 트리뷰용.
- graph-level `metadata_props` 는 비어 있음(전부 node-level).

전체 노드(`MaxPool`, `Reshape`, `Gemm`×2, `Relu`)에서 동일하게 5종이 채워짐. 원문: `artifacts/export_metadata.log`, 상세 리포트: `artifacts/onnx_export_*_success.md`.

### 결과 B: `dynamo=False` (torch 2.14) — 아무것도 안 남음

```
[node] /pool/MaxPool (MaxPool)  doc_string=''  metadata_props=0개
[node] /Flatten     (Flatten)  doc_string=''  metadata_props=0개
... (전 노드 동일)
```

### 결과 C: 리포의 `docs/mnist_example/mnist.onnx` (torch **1.12.0** 구식 export) — node `doc_string`에 스택트레이스 존재

이전 세션에서 "`doc_string`이 비어 있었다"고 기록되었으나, 그것은 **model-level** `doc_string`이다. **node-level** `doc_string`은 채워져 있다:

```
MaxPool_0  doc_string = "/usr/.../torch/nn/functional.py(782): _max_pool2d\n
                         ./gen_mnist_onnx.py(24): forward\n ... (전체 파이썬 콜스택)"
Gemm_3     doc_string = "/usr/.../torch/nn/modules/linear.py(114): forward\n
                         ./gen_mnist_onnx.py(26): forward\n ..."
```

→ **구식 exporter의 `strip_doc_string` 동작은 torch 버전에 의존적**이다. torch 1.12 시절에는 node `doc_string`에 스택트레이스를 남겼고, torch 2.14 legacy 경로는 남기지 않는다. 신식(`dynamo=True`)은 `metadata_props`로 항상 남긴다.

### 1순위 결론

- **신식 exporter를 쓰면 provenance는 풍부하게 확보된다** (`metadata_props`, 특히 `stack_trace`).
- 구식 exporter는 torch 버전에 따라 `doc_string`에 있을 수도/없을 수도 → 신식 경로를 표준으로 잡는 것이 안전.
- 어느 경우든 정보는 **ONNX 파일 안에 이미 존재**한다. 문제는 그 다음 단계(onnx-mlir)다.

---

## 2순위 — `metadata_props`가 onnx-mlir 임포터에서 살아남는가

### 소스 코드 근거 (결정적)

`src/Builder/FrontendDialectTransformer.cpp` — ONNX → MLIR 임포터:

```cpp
Location ImportLoc(const onnx::NodeProto &node) {
  if (node.has_name()) {
    return NameLoc::get(builder_.getStringAttr(node.name()));   // node.name() 만 사용
  } else {
    return UnknownLoc();
  }
}
```

```cpp
// 속성 임포트: node.attribute() 순회 + node.name() → "onnx_node_name"
if (node.has_name())
  attributes.push_back(builder_.getNamedAttr(
      "onnx_node_name", builder_.getStringAttr(node.name())));
```

- `grep -rn "metadata_props\|doc_string" src/` → **매치 0건**.
- 임포터가 읽는 것은 `node.attribute()`(ONNX op 속성), `node.name()`, `node.input/output` 뿐.
- `node.metadata_props()` / `node.doc_string()` 는 **한 번도 접근하지 않는다.**

### 실측

`build/Debug/bin/onnx-mlir --EmitONNXIR --mlir-print-debuginfo --preserveLocations tinycnn_dynamo.onnx`:

```mlir
%5 = "onnx.MaxPoolSingleOut"(%arg0) <{...}> {onnx_node_name = "node_max_pool2d"} : (...) loc(#loc6)
%7 = "onnx.Gemm"(%6, %0, %1) <{...}>  {onnx_node_name = "node_linear"}       : (...) loc(#loc8)
...
#loc6  = loc("node_max_pool2d")
#loc8  = loc("node_linear")
#loc9  = loc("node_relu")
```

- `loc()` = ONNX **node 이름** 문자열뿐. 파일/라인 없음.
- 추가 attribute도 `onnx_node_name`(= 같은 node 이름) 하나뿐. `namespace`, `stack_trace`, `class_hierarchy` 등은 **어디에도 없다.**
- `mnist.onnx`도 동일: `loc("MaxPool_0")`, `loc("Gemm_3")` — node `doc_string`의 스택트레이스는 전부 소실.

### 2순위 결론

**`metadata_props`(및 `doc_string`)는 onnx-mlir 임포터에서 통째로 버려진다.** 살아남는 provenance는 `node.name` 하나뿐이고, 그것도 "Conv_0" 같은 연산 이름 수준이다.

→ **admit 단계는 "옮겨담기"로 단순화할 수 없다.** 다음 중 하나가 필요:

1. **원본 ONNX를 직접 파싱** — admit 단계에서 `onnx.load()` 후 각 `node.name` ↔ `node.metadata_props['pkg.torch.onnx.stack_trace']` 를 뽑아 별도 **locmap 파일**(JSON/sidecar)로 저장. onnx-mlir이 이미 보존하는 `onnx_node_name` 을 조인 키로 사용.
2. 또는 **onnx-mlir 임포터에 최소 패치** — `ImportLoc()` 에서 `node.metadata_props()` 를 읽어 `FusedLoc` 또는 `NameLoc(FileLineColLoc(...))` 로 붙이기. (업스트림 포크가 필요하지만 30~50줄 규모. `onnx_node_name` 을 다는 코드 바로 옆.)

옵션 1이 "중간 컴파일러를 안 건드린다"는 제약과 맞다. locmap은 tail 단계까지 `onnx_node_name` 으로 계속 조인 가능.

---

## 3순위 — lowering 단계에서 provenance 생존 여부

### `loc()` 전파: `--EmitMLIR` (krnl/affine/memref)

```mlir
#loc1  = loc("node_max_pool2d")
#loc10 = loc("onnx.MaxPoolSingleOut"(#loc1))          // NameLoc 중첩 (op타입(원본노드명))
...
memref.alloc() ...                         loc(#loc10)
affine.store %cst_0, %alloca[] : memref<f32> loc(#loc10)
arith.maxnumf %8, %7 : f32                  loc(#loc10)
affine.store %9, %alloc[...]                loc(#loc10)   // 원본 1개 노드 → 수십 개 op, 전부 #loc10 태그
```

- 원본 ONNX 노드 1개가 affine loop nest 수십 op로 확장되어도, **모든 파생 op가 원본 노드로 back-reference** 되는 `loc`을 유지한다.
- 형태는 `FusedLoc`이 아니라 **`NameLoc` 중첩**: `loc("onnx.Gemm"(loc("node_linear")))`. 최상위 이름은 lowering을 만든 ONNX op 타입, 안쪽이 원본 노드명.

### `loc()` 전파: `--EmitLLVMIR` (LLVM dialect)

```
#loc5  = loc("node_max_pool2d")
#loc13 = loc("onnx.MaxPoolSingleOut"(#loc5))
```

- LLVM dialect 수준에서도 named `loc` **유지**. 계측 결과 llvm op 약 174개가 노드명 loc, 약 269개가 `loc(unknown)`(엔트리포인트 래퍼·memref 디스크립터 처리·글로벌 ctor 등 보일러플레이트).
- `DISubprogram`, `DICompileUnit`, `llvm.dbg`, `DILocation` → **0건.** 즉 MLIR 레벨 `NameLoc`은 있지만 LLVM debug-info attribute로는 전혀 낮춰지지 않는다.

### 정식 DWARF 경로: `--enable-debug-info` → **실패**

onnx-mlir에는 `--enable-debug-info` 플래그가 있고 (`src/Compiler/CompilerOptions.cpp:740`), 켜면
`mlir::LLVM::createDIScopeForLLVMFuncOpPass()` 가 추가된다 (`src/Compiler/CompilerPasses.cpp:569`).

```
$ onnx-mlir --EmitObj --enable-debug-info --preserveLocations tinycnn_dynamo.onnx
[2/5] Compiling and Optimizing MLIR Module
inlinable function call in a function with a DISubprogram location must have a debug location   ← 에러
```

- `tinycnn_dynamo.onnx`, `mnist.onnx` **둘 다** 같은 지점에서 실패.
- `--EmitLLVMIR --enable-debug-info` 도 동일 — MLIR을 뽑기도 전에 verifier에서 막힘.
- 원인: DIScope 패스가 함수에 `DISubprogram`을 붙이는데, 함수 내부의 런타임 헬퍼 호출(`omTensor*`, memref alloc 등)에 debug location이 없어서 LLVM dialect verifier가 거부. 즉 **onnx-mlir이 생성하는 IR 형태와 이 stock 패스가 호환되지 않는다.**

### 완성된 `.so`의 DWARF 출처

```
$ onnx-mlir --EmitLib tinycnn_dynamo.onnx -o tc_lib      # (--enable-debug-info 없이)
$ readelf -S tc_lib.so | grep debug
  .debug_info .debug_line .debug_abbrev .debug_str ...    ← 섹션이 있다!
$ llvm-dwarfdump --debug-info tc_lib.so | grep producer
  DW_AT_producer ("GNU C17 13.3.0 ... -g ...")
  DW_AT_name     ("/workdir/onnx-mlir/src/Runtime/OMTensor.c")
$ llvm-dwarfdump --debug-line tc_lib.so | grep name:
  "OMTensor.c" "OMTensorList.c" "stdio.h" ...             ← 전부 C 런타임 소스
```

- `.so`에 DWARF가 있지만 **전부 onnx-mlir C 런타임**(`OMTensor.c`, `OMTensorList.c` 등, `-g`로 컴파일된 정적 라이브러리)에서 온 것.
- 모델 연산 함수(`main_graph_tc_lib`, lowering된 MaxPool/Gemm 루프)에는 **`.debug_line` 엔트리가 없다.** gdb로 이 함수에 들어가면 소스 매핑 불가.

### 참고: 이미 존재하는 수동 DWARF 생성 코드

`src/Conversion/KrnlToLLVM/KrnlInstrument.cpp` 는 **명령어 프로파일링용**으로 `DICompileUnitAttr` + `DISubprogramAttr` + `FusedLocWith<DISubprogramAttr>` 를 손으로 만든다 (업스트림 PR #3523, #3552). `--instrument-*` 옵션을 켤 때만 동작하고 각 instrument call site에 대해서만 DWARF를 붙이지만, **"generic 패스를 고치지 않고, 대상 op에만 손으로 DWARF를 심는" 접근이 onnx-mlir 안에서 이미 검증되어 있다**는 선례로서 의미가 있다. 우리 디버거용 provenance 주입도 같은 패턴을 쓸 수 있다.

### 3순위 결론

- **좋은 소식**: `onnx_node_name` 기반 `loc()`는 LLVM dialect까지 **손실 없이 전파**된다. 원본 노드 → 파생 op N개 back-reference가 성립.
- **나쁜 소식 1**: 그 `loc`은 **노드 이름 문자열일 뿐** 소스 파일·라인이 아니다 (1순위·2순위에서 이미 유실됐기 때문).
- **나쁜 소식 2**: MLIR `loc` → 정식 LLVM/DWARF 로 낮추는 경로(`--enable-debug-info`)가 **현재 업스트림에서 깨져 있다.** 모델 코드용 `.debug_line` 을 얻을 방법이 기본 제공되지 않는다.
- → **`#line` 주입(tail 단계) 우회 설계의 정당성이 실측으로 재확인됨.** EmitC→C 경로는 애초에 LLVM dialect를 안 거치므로 위 "나쁜 소식 2"와도 무관하고, C 텍스트에 `#line file lineno` 만 심으면 표준 `gcc -g`/DWARF/gdb 가 그대로 동작한다.

---

## Debugger 연계 협의용 정리 (입력 대상 / 관측 지점 / 출력 형식)

| 구분 | 내용 |
|---|---|
| **입력 대상** | `torch.onnx.export(..., dynamo=True)` 산출 ONNX. node별 `metadata_props`에 `pkg.torch.onnx.stack_trace`(파일·라인·소스라인), `namespace`/`class_hierarchy`/`name_scopes`(모듈 계층), `pkg.torch.onnx.fx_node`(FX 표현) 존재. 구식 export는 신뢰 불가(torch 버전 의존). |
| **관측 지점 1 (admit)** | ONNX 로드 직후. onnx-mlir이 `metadata_props`를 버리기 전에 우리가 직접 캡처. 조인 키 = `node.name`(= onnx-mlir의 `onnx_node_name` attr). 출력: `locmap` sidecar (예: `{"node_linear": {"file": "...", "line": 23, "src": "x = self.relu(self.fc1(x))", "scope": "TinyCNN.fc1"}}`). |
| **관측 지점 2 (ONNXIR/MLIR)** | `--EmitONNXIR --mlir-print-debuginfo` 출력의 `loc("<node_name>")` + `{onnx_node_name}` attr. 여기까지는 노드 이름만 보존. locmap과 조인하면 소스 위치 복원 가능. `--EmitMLIR`/`--EmitLLVMIR`까지 `loc`이 파생 op에 전파되므로 조인은 그 이후 단계에서도 유효. |
| **관측 지점 3 (tail, EmitC→C)** | 생성된 C 코드. 각 연산 블록 앞에 `#line <lineno> "<원본 .py>"` 주입 (locmap + `onnx_node_name` 주석 매칭). 이후 표준 `gcc -g` → DWARF `.debug_line` → gdb 중단점/스텝. `readelf --debug-dump=decodedline` 로 검증 가능(별도 환경에서 실증 완료). |
| **출력 형식** | (a) locmap sidecar(JSON): `onnx_node_name` → 소스위치·모듈계층. (b) `#line`이 박힌 C. (c) 표준 DWARF가 실린 host-mode `.so`. 디버거는 (c)를 gdb 호환으로 직접, 또는 (a)를 통해 ONNX 노드 단위 심볼릭 뷰 제공. |
| **onnx-mlir 정식 DWARF (`--enable-debug-info`)** | 현재 버전에서 **미동작**(verifier 에러). 협의 시 "모델 코드용 DWARF는 onnx-mlir 기본 경로로는 안 나온다, `#line` 우회 또는 KrnlInstrument식 수동 DI 주입이 필요"로 공유. |

---

## 다음 작업 후보 (이번 조사 범위 밖)

1. admit 단계 locmap 캡처 PoC: `metadata_props` → sidecar JSON 생성기 (파이썬, 중간 컴파일러 불건드림).
2. `--enable-debug-info` verifier 실패의 정확한 위치 추적 (`--mlir-print-ir-after-failure`) — stock 패스 앞에 debug-loc 채우는 프리패스로 살릴 수 있는지 판단.
3. tail 단계: 실제 EmitC→C 산출물에 `#line` 주입 스크립트 + `readelf --debug-dump=decodedline` 회귀 테스트. (이 컨테이너는 EmitC 경로 없음 → torch/ARX 있는 환경 필요.)
4. `onnx_node_name` 이 `--EmitLib` 산출 `.so` 심볼/`OMCurrentNodeName` 런타임 변수와 어떻게 연결되는지 (instrument 경로 재사용 가능성).

---

## 부록: 명령/파일 목록

| 파일 | 내용 |
|---|---|
| `run_investigation.sh` | 전체 재현 스크립트 |
| `artifacts/export_metadata.log` | 1순위 `inspect_export_metadata.py` 원본 출력 |
| `artifacts/dynamo_exporter_report.md` | dynamo exporter 자체 리포트 (재실행 시 `onnx_export_*_success.md` 로 새로 생성됨) |
| `artifacts/tinycnn_dynamo.onnx`, `tinycnn_legacy.onnx` | 실험용 ONNX (신식/구식) |
| `artifacts/tinycnn_dynamo_onnxir.onnx.mlir` | `--EmitONNXIR --mlir-print-debuginfo` 출력 |
| `artifacts/tinycnn_dynamo_onnxir.locs.txt` | 위 파일에서 op·loc만 추출 |
| `artifacts/tinycnn_dynamo_mlir.onnx.mlir` | `--EmitMLIR` (krnl/affine) 출력 |
| `artifacts/tinycnn_dynamo_llvm.onnx.mlir` | `--EmitLLVMIR` (LLVM dialect) 출력 |
| `artifacts/full_run.log` | `run_investigation.sh` 실행 로그 |

핵심 명령:

```bash
OM=build/Debug/bin/onnx-mlir
$OM --EmitONNXIR --mlir-print-debuginfo --preserveLocations model.onnx -o out
$OM --EmitMLIR   --mlir-print-debuginfo --preserveLocations model.onnx -o out
$OM --EmitLLVMIR --mlir-print-debuginfo --preserveLocations model.onnx -o out
$OM --EmitObj    --enable-debug-info    --preserveLocations model.onnx -o out   # ← verifier 에러
$OM --EmitLib model.onnx -o out.so ; readelf -S out.so | grep debug
```

---

# 2부 — 도구 구현: locmap 캡처 + `#line` 투영 (TASK_implement_dwarf_pipeline_v2)

- 작성일: 2026-09-02 (같은 세션 후반)
- 구현물:
  - `debug_info_investigation/locmap_capture.py` — Part A (admit 단계)
  - `debug_info_investigation/inject_line_directives.py` — Part B (tail 단계)
  - `debug_info_investigation/run_pipeline.sh` — A→조인확인→A2진단→B→재현성 일괄 재현
- 검증 대상: `artifacts/tinycnn_dynamo.onnx` (torch 2.14.0+cpu, `dynamo=True`, opset 20, IR v10)

## Part A — admit 단계 locmap 캡처 (완료)

### 무엇을 하나
onnx-mlir 임포터가 `metadata_props`/`doc_string` 을 버리기 전에(1부 2순위 결론), 원본 ONNX 를
직접 읽어 **`node.name` → 소스 위치** 매핑을 sidecar JSON 으로 떠 둔다. 조인 키는
`node.name` == onnx-mlir 의 `onnx_node_name` 속성.

### 결과 (`artifacts/tinycnn_dynamo.locmap.json`)
- 5개 노드 전부(`node_max_pool2d`, `node_view`, `node_linear`, `node_relu`, `node_linear_1`) 위치 매핑 성공 (5/5).
- 각 노드 엔트리: `file`, `line`, `source`(소스 한 줄), `func`, `module_scope`(`name_scopes` 유래, 예 `fc1.linear`),
  `namespace`(torch 원문), `class_hierarchy`, `fx_node`, `stack_frames`(사용자+torch 전체 콜스택).
- `pkg.torch.onnx.stack_trace` 를 표준 파이썬 트레이스백으로 파싱 → **프레임워크 내부(`site-packages`/`dist-packages`)가
  아닌 첫 프레임**을 중단점 대상(`file`/`line`)으로 선택. 예: `node_linear` → `inspect_export_metadata.py:23`,
  `x = self.relu(self.fc1(x))`.

### 조인 키 검증
`onnx-mlir --EmitONNXIR` 산출물의 `onnx_node_name` 속성 5개 집합 == locmap 키 5개 집합 (`run_pipeline.sh` 에서 `diff` 로 자동 확인, 일치).

### 결정성
같은 `.onnx` 입력 → locmap JSON 바이트 단위 동일 (3회 반복 sha256 일치). 타임스탬프/난수 없음.
JSON 키 순서 = 그래프 노드 순서(정보이므로 `sort_keys` 미사용), `ensure_ascii=False`, 마지막 개행 1개.

## Part A2 — 진짜 MLIR→LLVM DWARF 경로(`--enable-debug-info`) 복구 시도 (타임박스, **실패로 기록**)

### 진단 결과 (상세: `artifacts/partA2_enable_debug_info_diagnosis.txt`)
- **실패 pass**: `DIScopeForLLVMFuncOpPass` (`ensure-debug-info-scope-on-llvm-func`) — stock 업스트림 MLIR 패스.
- **에러**: `inlinable function call in a function with a DISubprogram location must have a debug location`
  (MLIR LLVM-dialect verifier). `tinycnn_dynamo.onnx`, `mnist.onnx` 둘 다 같은 지점에서 실패.
- **원인 (확인됨)**: 이 패스는 **본문이 있는 모든 `llvm.func` 에 `DISubprogram` 을 붙인다** — onnx-mlir 이
  생성한 래퍼/쿼리 함수(`omInputSignature`, `omOutputSignature`, `omQueryEntryPoints`, `run_main_graph`,
  `_mlir_ciface_main_graph`, `omTensorPrint` 헬퍼 등)와 `main_graph` 포함. 그런데 이 함수들 안의 수많은
  `llvm.call`(`@malloc`, `@free`, `@omTensorListGetSize`, `@printf`, `@strncmp`, `@omTensorGetShape`, …)이
  `loc(unknown)` 이다. verifier 규칙("`DISubprogram` 스코프 함수 안의 inlinable call 은 debug location 필수")에
  걸린다. 즉 **한 op 문제가 아니라, onnx-mlir 이 런타임 헬퍼 호출을 전부 `UnknownLoc` 으로 만들기 때문의 구조적 문제.**

### 왜 여기서 중단했나 (타임박스 준수)
- 원인 파악은 타임박스(1~2시간) 내에 끝났으나, **고치려면**: (a) `src/` 에 새 MLIR 패스(“`DISubprogram` 스코프
  함수 안의 `UnknownLoc` op 에 함수 `DISubprogram` 위치를 fallback 으로 채우는 프리패스”, `KrnlInstrument.cpp` 의
  `FusedLocWith<LLVM::DISubprogramAttr>` 패턴 참고)를 추가하고, (b) `CompilerPasses.cpp` 의 패스 순서에 끼워넣고,
  (c) **onnx-mlir 전체 재빌드(Debug)** 가 필요하다. 이는 "여러 pass 를 건드려야 하는 게 명확"한 경우이자
  대규모 빌드라, TASK v2 의 타임박스 지침(“그 시점에서 바로 중단, Part B 로 진행”)과 CLAUDE.md 의 빌드 주의사항에 따라
  **소스 수정 없이 진단만 기록하고 중단**했다. (onnx-mlir/`third_party` 는 한 줄도 수정하지 않았다.)
- **부차적 근거 — 고쳐도 이득이 작다**: `--EmitLLVMIR` 산출물에서 모델 op 의 `loc` 은 전부 `NameLoc`
  (`loc("onnx.Gemm"(loc("node_linear")))`)이지 `FileLineColLoc` 이 아니다. `DIScopeForLLVMFuncOpPass` 는
  `FileLineColLoc` 이 아닌 위치를 line 0 으로 떨어뜨리므로, verifier 를 통과시켜도 **모델 코드의 `.debug_line`
  엔트리는 line 0** 이 된다. 진짜 소스 라인을 넣으려면 결국 locmap 을 MLIR 단계에서 `FileLineColLoc` 로 주입하는
  별도 작업이 필요하다 — 이건 `--enable-debug-info` 복구와는 독립적이며 Part A 의 locmap 이 그 입력이 된다.

**결론**: Part A2 = 실패(원인은 stock 패스 + onnx-mlir 의 pervasive `UnknownLoc`, 소스 수정+재빌드 필요 → 타임박스 초과로 중단). Part B 로 진행함.

## Part B — tail 단계 `#line` 투영 (완료, readelf/gdb 로 실증)

### 무엇을 하나
`inject_line_directives.py` 가 생성 C 코드의 **노드 마커 주석**(`/* onnx_node_name: <name> */`, EmitC 가
`onnx_node_name` 속성을 주석으로 내보내면 그대로 나올 형태)을 찾아, 그 직후에 `#line <L> "<원본.py>"` 를
삽입한다. 노드 블록이 끝나면(`/* onnx_node_end */` 또는 다음 마커) `#line` 으로 다시 C 파일에 매핑을 되돌린다.
`<L>`/`<원본.py>` 는 Part A 의 locmap 에서 `onnx_node_name` 으로 조인해 가져온다.

이 컨테이너에는 실제 EmitC→C 경로가 없으므로, `--demo` 모드가 locmap 으로부터 **EmitC 풍 C**(노드마다 마커 +
작은 커널 함수)를 만들어 투영→`gcc -g -O0`→검증까지 한 번에 수행한다.

### 검증 결과 (`artifacts/partB/tinycnn_dynamo_partB_report.json`, `run_pipeline.log`) — 4/4 PASS
1. `gcc -g` 가 `.debug_line` 섹션을 생성함.
2. `.debug_line` 에 원본 `.py` 파일명(`inspect_export_metadata.py`)이 등장함.
3. locmap 이 투영한 `(inspect_export_metadata.py, {21,22,23,24})` 네 조합이 **DWARF 라인테이블에 그대로** 존재함
   (`readelf --debug-dump=decodedline` 파싱). 즉 MaxPool 노드 코드의 PC 들이 `.py:21`, Gemm 노드가 `.py:23`,
   ... 로 매핑됨.
4. `gdb --batch -ex "break inspect_export_metadata.py:21"` →
   `Breakpoint 1 at 0x1159: file .../inspect_export_metadata.py, line 21`, 그 주소가 `<node_node_max_pool2d+16>`.
   **원본 PyTorch 소스 줄에 건 중단점이 해당 ONNX 노드의 컴파일된 코드 안에 바인딩된다** — tail 단계 목표 그대로.

### 결정성
`_kernels.c`(생성)와 `_kernels.line.c`(투영) 모두 3회 반복 실행에서 sha256 동일. 일반 모드(`inject_line_directives.py IN.c locmap.json`) 출력도 3회 동일. 순수 텍스트 변환, 타임스탬프/난수 없음.
(`gcc` 산출 `.o`/실행파일의 바이트 동일성은 보장 대상 아님 — 검증은 우리 스크립트의 텍스트 산출물 기준.)

## 재현성 체크 (DoD 항목) — 통과

| 스크립트 | 반복 | 결과 |
|---|---|---|
| `locmap_capture.py` (locmap JSON) | 3회 | sha256 동일 |
| `inject_line_directives.py --demo` (`_kernels.c`, `_kernels.line.c`) | 3회 | sha256 동일 |
| `inject_line_directives.py` 일반 모드 (투영 C) | 3회 | sha256 동일 |

`run_pipeline.sh` 가 매 실행 끝에 2회 반복 후 `cmp` 로 자동 확인한다.

## RTST(디버거) 검증 요청 항목 매핑

| 요청 항목 | 이 작업으로 충족되는 근거 |
|---|---|
| **AI 포맷 로드** | `locmap_capture.py` 가 `dynamo=True` 로 export 한 ONNX 를 `onnx.load()` 로 실제 로드하여 그래프의 모든 node 를 순회하고, node 별 `metadata_props`(특히 `pkg.torch.onnx.stack_trace`)에서 디버깅 정보를 추출해 locmap 으로 낸다. `tinycnn_dynamo.onnx` 5/5 노드에서 성공. 즉 "AI 포맷(ONNX)을 로드해 디버깅 정보를 뽑는다"가 실동작으로 충족됨. |
| **레이어별 정보 측정** | 확인 대상 정보의 초안 = locmap 스키마: `file`, `line`, `source`, `func`, `module_scope`(예 `fc1.linear`), `class_hierarchy`(`['__main__.TinyCNN','torch.nn.modules.linear.Linear','aten.linear.default']`), `fx_node`, `stack_frames`. 노드(=레이어) 단위로 이 표를 제공한다. 아래 "locmap 스키마" 표를 제안 스펙으로 사용. |
| **DNN 연산자 지원** | locmap 은 `node.name` 기반이라 **연산자 종류에 대한 분기가 전혀 없다**. MaxPool/Reshape/Gemm/Relu 를 동일 코드 경로로 커버함(검증됨). 새 연산자 추가 시 도구 수정 불필요 — 설계상 이미 만족. |
| **동일 입력 일관된 출력 / 반복 재현성** | 위 "재현성 체크" 표 참고. `locmap_capture.py`, `inject_line_directives.py` 모두 같은 입력에 3회 바이트 동일 출력. |

관련 없음(범위 밖, 손대지 않음): SNN 레이어별 발화율 측정, SNN 타임스텝 지원 — SNN 담당(UNIST) 몫.

### locmap 스키마 (레이어별 정보 측정 항목의 제안 스펙)

| 필드 | 예시 | 의미 / 디버거 용도 |
|---|---|---|
| `op_type` | `Gemm` | ONNX 연산자 종류 |
| `file` | `inspect_export_metadata.py` | 중단점 대상 소스 파일 |
| `line` | `23` | 중단점 대상 라인 |
| `source` | `x = self.relu(self.fc1(x))` | 소스 코드 한 줄(표시용) |
| `func` | `forward` | 소스 함수명 |
| `module_scope` | `fc1.linear` | PyTorch 모듈 경로(디버거 트리뷰/네임스코프) |
| `namespace` | `: __main__.TinyCNN/fc1: ...Linear/linear: aten.linear.default` | torch 원문 계층 문자열 |
| `class_hierarchy` | `["__main__.TinyCNN", "...Linear", "aten.linear.default"]` | 클래스 계층(리스트) |
| `fx_node` | `%linear : ... = call_function[target=torch.ops.aten.linear.default](...)` | 원본 FX 노드 표현(변수 추적 힌트) |
| `stack_frames` | `[{file,line,func,source}, ...]` | 사용자+프레임워크 전체 콜스택(상위 프레임 탐색용) |
| `_meta.join_key` | `onnx_node_name` | onnx-mlir 산출물과의 조인 키 |
| `_meta.onnx_sha256` | `307e6d...` | 어느 ONNX 로 뜬 locmap 인지 고정 |

## 갱신된 Definition of Done 대조

- [x] Part A: `locmap_capture.py` 완료, `tinycnn_dynamo.onnx` 로 검증 (5/5 노드, 조인 키 일치)
- [x] Part A2: 성공/실패와 원인 기록 (실패 — stock `DIScopeForLLVMFuncOpPass` + pervasive `UnknownLoc`; 타임박스 준수, 소스 미수정)
- [x] Part B: `inject_line_directives.py` 완료, 원본 `.py:line` 이 DWARF 에 찍히고 gdb 중단점이 노드 코드에 바인딩됨을 readelf/gdb 로 확인
- [x] 재현성 체크: A/B 스크립트 3회 반복 바이트 동일
- [x] RTST 매핑: "AI 포맷 로드" / "레이어별 정보 측정" 근거 기록 (+ 스키마 표)
- [x] `debug_info_investigation.md` 에 이어붙임 (1부 내용 보존)

## 남은 일 / 실제 ARX 파이프라인 적용 시 필요한 것

1. **실제 EmitC 출력 대상 테스트**: 이 컨테이너엔 EmitC 경로가 없어 데모 C 로 실증했다. PNU harx/EmitC 산출
   C 에 `onnx_node_name` 이 주석 등으로 남는지 확인하고, 남지 않으면 EmitC emitter 쪽에 노드명 주석을 추가하도록
   협의 필요(우리가 tail 단계에서 소유). 마커 정규식은 `--marker-re` 없이도 관대하게 잡도록 되어 있으나 실제 포맷 확인 필요.
2. **admit 단계 위치**: `locmap_capture.py` 를 admit 파이프라인에 endpoint 로 붙이고, 산출 locmap 을 tail 까지 전달하는 경로 확정.
3. **`#line` 세분도**: 데모는 노드=함수 1:1 이라 함수 단위로 투영했다. 실제 EmitC 는 한 노드가 여러 문장으로 펼쳐질 수 있으니,
   노드 시작 지점에서 한 번 `#line` 을 걸고 그 뒤 문장들이 모두 같은 `.py:line` 으로 접히는 현재 방식이 적절한지(혹은
   노드 내부를 더 잘게 나눌지) 디버거 팀과 합의.
4. **(선택) MLIR 단계 `FileLineColLoc` 주입**: onnx-mlir CPU 실행을 "레퍼런스 실행"으로 쓰려면, locmap 을
   `--EmitMLIR` 후 `loc` 에 `FileLineColLoc` 로 심는 별도 패스가 필요(Part A2 참고). 우선순위 낮음.

## 부록: 명령

```bash
cd debug_info_investigation
# 전체 파이프라인 (A → 조인확인 → A2진단 → B → 재현성)
./run_pipeline.sh                         # 기본 artifacts/tinycnn_dynamo.onnx
./run_pipeline.sh /path/to/other_dynamo.onnx

# 개별
python3 locmap_capture.py MODEL.onnx --relative-to /workdir/onnx-mlir -o MODEL.locmap.json
python3 inject_line_directives.py kernels.c MODEL.locmap.json -o kernels.line.c   # 일반
python3 inject_line_directives.py --demo MODEL.locmap.json --workdir ./artifacts/partB  # 데모+검증
```

---

# 3부 — 구조 정리 (v3, TASK_formalize_architecture_v3)

- 작성일: 2026-09-03
- **성격**: 순수 리팩터링. v2의 결론(Part A/B 성공, A2 실패)을 재검증하지 않는다. 동작하는 데모 스크립트를,
  실제 harx/EmitC가 연결될 때 한 부분만 갈아끼우면 되는 모듈 구조로 정리한 것.
- 산출물: `debug_info_investigation/artifacts/v3/`

## 확정된 아키텍처 (코드에 그대로 반영됨)

```
                 [ONNX 파일 — metadata_props 포함, 어느 쪽도 안 건드림]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      정보 트랙 (우리 코드)          코드 트랙 (외부, 건드리지 않음 — 지금은 데모)
   info_track.build_locmap()        code_track.get_generated_c()
   metadata_props 읽음               node.name / op_type 만 읽음
              │                           │
              ▼                           ▼
        locmap dict                  생성된 C 소스 문자열
   node.name → 원본 .py 파일:줄        node.name 마커만 남음
              │                           │
              └─────────────┬─────────────┘
                            ▼
                    join.inject()
              (조인 키 = node.name, #line 삽입)
                            │
                            ▼
                   gcc -g → 실행파일
              (진짜 DWARF, gdb로 원본 .py 줄 중단점)
```

## 모듈 구성

| 파일 | 공개 API | 역할 | 트랙 독립성 |
|---|---|---|---|
| `info_track.py` | `build_locmap(onnx_path, *, relative_to=None) -> dict`, `dumps_deterministic(locmap) -> str` | ONNX `metadata_props` → locmap | 코드 트랙 존재 몰라도 됨 |
| `code_track.py` | `get_generated_c(onnx_path) -> str` | 코드 컴파일 경로의 **대역**. ONNX 그래프에서 `node.name`/`op_type`만 읽어 마커 붙은 데모 C 생성 | 정보 트랙·locmap·원본 .py 위치 몰라도 됨 |
| `join.py` | `inject(c_source, locmap) -> str` (+ `inject_with_stats`, `find_node_markers`) | 두 트랙이 만나는 **유일한 지점**. 마커 ↔ locmap 조인(키=`node.name`), `#line` 삽입 | C 문자열 + dict만 받음, 두 트랙 모듈 import 안 함 |
| `run_full_pipeline.py` | `run(onnx_path, workdir, relative_to) -> dict`, `main()` | 오케스트레이터: build_locmap → get_generated_c → inject → `gcc -g` → readelf/gdb 검증 → 3회 재현성 | — |
| `locmap_capture.py` | (shim) | v2 이름 하위호환. `info_track.main` 재노출 | — |
| `inject_line_directives.py` | (shim) | v2 이름 하위호환. 일반 모드→`join.main`, `--demo`→`run_full_pipeline.run` | — |
| `run_pipeline.sh` | — | `run_full_pipeline.py`를 부르는 얇은 셸 (+ 조인 키 사전점검) | — |

### 실제 harx/EmitC 연동 지점 (명시)

`code_track.get_generated_c(onnx_path: str) -> str` **구현부만 교체**하면 된다.
- 입력: onnx 파일 경로 / 출력: C 소스 문자열 (노드 구획마다 `onnx_node_name`이 식별 가능하게 —
  `join.py`의 마커 정규식 `onnx[_-]node(_name)? [:=] <name>`에 걸리면 형식은 관대하게 인식됨).
- 시그니처(경로→C 문자열) 유지. `join.py` / `run_full_pipeline.py`는 안 건드림.
- 정보 트랙과 독립 유지: 이 함수는 locmap도 원본 .py 위치도 몰라야 한다.
- (`code_track.py` 모듈 docstring에 같은 내용 명시.)

## v2 대비 바뀐 것 / 안 바뀐 것

- **안 바뀜**: locmap 스키마·바이트(동일 sha256), `join`의 `#line` 계산 로직, 데모 C의 노드 구획 7줄 구조,
  검증 로직(readelf 파서 + gdb probe).
- **바뀜(구조/무해)**:
  - 데모 C가 v2에서는 주석에 `py-source: ...` / `module_scope=...`를 넣었다(–demo가 locmap을 갖고 있어서
    우연히 가능했던 것). v3의 `code_track`은 **provenance를 안 읽는다**는 확정 아키텍처에 맞춰 그 주석 조각을 뺐다.
    노드 구획의 **줄 수(7줄)는 그대로**라 `#line` 투영 결과(DWARF `.py` 매핑)는 불변.
  - 투영 C 헤더 문구 `by inject_line_directives.py` → `by join.py` (줄 수 3 유지).
  - 산출물 위치 `artifacts/partB/` → `artifacts/v3/`.

## 회귀 확인 결과 (필수 항목) — 통과, v2와 동일

| 항목 | v2 | v3 | 동일? |
|---|---|---|---|
| locmap JSON sha256 | `6de7b86…` | `6de7b86…` | ✅ 바이트 동일 |
| `readelf --debug-dump=decodedline` 의 `.py` (파일,라인) | `21,22,23,24,25` | `21,22,23,24,25` | ✅ |
| 투영한 `(inspect_export_metadata.py, {21,22,23,24})` DWARF 존재 | 4/4 | 4/4 | ✅ |
| `gdb break inspect_export_metadata.py:21` | `Breakpoint 1 at 0x1159: file /workdir/onnx-mlir/inspect_export_metadata.py, line 21.` (`<node_node_max_pool2d+16>`) | 문자열까지 동일 | ✅ |
| 검증 체크 통과 | 4/4 | 4/4 | ✅ |
| 3회 반복 바이트 동일 (locmap, 투영 C) | 통과 | 통과 | ✅ |
| 하위호환 shim (`locmap_capture.py`, `inject_line_directives.py` 일반/`--demo`) | — | 전부 동작, locmap sha 동일 | ✅ |

원문: `artifacts/v3/tinycnn_dynamo_pipeline_report.json`, `artifacts/v3/run_pipeline.log`.

## 부록: 명령 (v3)

```bash
cd debug_info_investigation
./run_pipeline.sh                              # 조인키 점검 + 전체 파이프라인 + 재현성
./run_pipeline.sh /path/to/other_dynamo.onnx

# 개별 모듈
python3 info_track.py MODEL.onnx --relative-to /workdir/onnx-mlir -o MODEL.locmap.json
python3 code_track.py MODEL.onnx -o kernels.c
python3 join.py kernels.c MODEL.locmap.json -o kernels.line.c --resolve-paths
python3 run_full_pipeline.py MODEL.onnx --workdir ./artifacts/v3

# 파이썬에서 직접
python3 -c "import info_track, code_track, join
lm = info_track.build_locmap('MODEL.onnx')
c  = code_track.get_generated_c('MODEL.onnx')
print(join.inject(c, lm))"
```
