# ONNX 디버깅 정보 보존 — 진행 정리 (Notion용)

> **과제**: 재구성형 AI 프로세서 SW 프레임워크 4세부 (디버거·프로파일러·IDE)
> **목표**: "ONNX은 원래 소스 위치 디버깅 정보를 남기지 않는다"는 문제를, MLIR/DWARF 방식을 참고해
> 실제 디버깅(중단점·변수 추적)이 가능하도록 설계
> **환경**: 업스트림 onnx-mlir `v0.5.1.0-69-g706eb3d6`, LLVM 23 / `torch 2.14.0+cpu`, `dynamo=True`
> **전제**: 이 도커는 PNU의 ARX/harx 컴파일러가 아니다. 아래 결론은 전부 "일반 MLIR/onnx-mlir 인프라 기준"이며 ARX 경로 재현이 아니다.
> **작성일**: 2026-09-07

---

## 0. 한 줄 요약

PyTorch `dynamo=True` export ONNX 는 노드별 `metadata_props` 에 원본 `.py` 파일·줄·스택트레이스를 다 담는다.
그러나 **onnx-mlir 임포터가 이걸 통째로 버린다**(살아남는 건 노드 이름뿐).
그래서 우리는 원본 ONNX 를 직접 읽어 **`onnx_node_name → 원본 .py:줄` 매핑(locmap sidecar)** 을 따로 뜨고,
tail 단계에서 생성 C 에 `#line` 으로 그 줄을 다시 꽂는다. `gcc -g` → DWARF → gdb 중단점까지 실제 바이너리로 실증 완료.

---

## 1. 배경과 문제

- 우리가 실제로 수정 가능한 지점은 **admit 단계(ONNX 로드 직후)** 와 **tail 단계(EmitC→C 이후)** 두 곳뿐. 중간 `mlir-arx` 컴파일러 내부는 절대 안 건드림.
- 실제 프로덕션 경로: `ONNX → onnx-mlir → harx(mlir-arx) → EmitC → C → host-mode 실행 아티팩트`.
- 이 도커에는 `harx`/`EmitC`/ARX 가 없어서 그 경로를 재현할 수 없음 → "같은 MLIR 인프라 위에서 provenance 가 얼마나 보존되는가"를 검증하는 **실험대**로 사용.

---

## 2. 핵심 발견 (조사 결과)

| 항목 | 결론 |
| --- | --- |
| PyTorch `dynamo=True` export 시 provenance | 노드별 `metadata_props` 에 원본 소스 파일·줄·스택트레이스·모듈 계층이 **전부** 들어간다 (`pkg.torch.onnx.stack_trace` 등 5종) |
| PyTorch 구식(`dynamo=False`) export | torch 2.14 에서는 노드 `doc_string`·`metadata_props` 모두 빈 값 (torch 1.12 시절엔 `doc_string` 에 스택트레이스가 있었음 — 버전 의존적) |
| onnx-mlir 임포터가 `metadata_props`/`doc_string` 을 읽는가 | **아니오. 완전히 버린다.** 소스에 참조 0건. `node.name()` 만 사용 |
| onnx-mlir 임포터가 보존하는 것 | `node.name()` → `loc("<name>")` (NameLoc) + `onnx_node_name` 속성. 그게 전부 (파일·줄 없음) |
| `loc()` 가 lowering 에서 살아남는가 | 살아남는다. `--EmitONNXIR` → `--EmitMLIR` → `--EmitLLVMIR` 까지 모든 파생 op 에 `NameLoc` 중첩으로 전파. `FusedLoc` 아님, 소실 아님. 단 여전히 "이름 문자열"일 뿐 |
| 정식 DWARF 생성 (`--enable-debug-info`) | **깨져 있다.** stock `DIScopeForLLVMFuncOpPass` 가 MLIR LLVM verifier 에러(`inlinable function call ... must have a debug location`)로 실패. mnist·dynamo 모델 모두 |
| 완성된 `.so` 의 DWARF | 있긴 하나 onnx-mlir C 런타임(`OMTensor.c` 등)에서 온 것뿐. 모델 연산 코드(`main_graph`)엔 라인 정보 없음 |

**설계 함의**
- **admit 단계**: onnx-mlir 이 provenance 를 버리므로 "이미 있는 걸 옮겨담기"가 아니라 **우리만의 locmap 캡처 로직이 필요**하다.
- **tail 단계**: 정식 DWARF 경로가 동작하지 않으므로, 생성 C 에 `#line` 지시자를 심는 우회 설계가 (현 업스트림 기준) 유일하게 현실적인 선택지.

---

## 3. 확정 아키텍처 (2-트랙)

```
                 [ONNX 파일 — metadata_props 포함, 어느 쪽도 안 건드림]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      정보 트랙 (우리 코드)          코드 트랙 (외부, 지금은 데모 대역)
   info_track.build_locmap()        code_track.get_generated_c()
   metadata_props 읽음               node.name / op_type 만 읽음
              │                           │
              ▼                           ▼
        locmap dict                  생성된 C 소스 문자열
   node.name → 원본 .py:줄            node.name 마커만 남음
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     join.inject()
              (조인 키 = node.name, #line 삽입)
                            │
                            ▼
                    gcc -g → 실행파일
              (진짜 DWARF, gdb 로 원본 .py 줄 중단점)
```

| 파일 | 공개 API | 역할 |
| --- | --- | --- |
| `info_track.py` | `build_locmap(onnx_path, *, relative_to=None) -> dict` | ONNX `metadata_props` → locmap. 코드 트랙 존재 몰라도 됨 |
| `code_track.py` | `get_generated_c(onnx_path) -> str` | 컴파일 경로의 **대역**. `node.name`/`op_type` 만 읽어 마커 붙은 데모 C 생성. **실제 harx/EmitC 연동 시 이 함수 본문만 교체** |
| `join.py` | `inject(c_source, locmap) -> str` | 두 트랙이 만나는 **유일한 지점**. 마커 ↔ locmap 조인(키 = `node.name`), `#line` 삽입 |
| `run_full_pipeline.py` | `run(onnx_path, workdir, relative_to) -> dict` | 오케스트레이터: build_locmap → get_generated_c → inject → `gcc -g` → readelf/gdb 검증 → 3회 재현성 |
| `locmap_capture.py` / `inject_line_directives.py` | (shim) | v2 이름 하위호환 |
| `run_pipeline.sh` | — | `run_full_pipeline.py` 를 부르는 얇은 셸 (+ 조인 키 사전점검) |

---

## 4. 원본 Python 줄은 어떻게 남는가

### 4.1 어디에 남나

| 경로 | 원본 `.py:줄` 보존? | 형태 |
| --- | --- | --- |
| onnx-mlir 임포터 (`--EmitONNXIR` 등) | ❌ | `loc("node_linear")` — 노드 이름 문자열뿐 |
| **info_track locmap (우리)** | ✅ | `{"node_linear": {"file": "model.py", "line": 23, ...}}` sidecar JSON |
| lowering (`--EmitMLIR` / `--EmitLLVMIR`) | ❌ (줄은 여전히 없음) | `loc("onnx.Gemm"(loc("node_linear")))` — 파생 op 전부가 노드 이름 back-ref |
| tail 생성 C → DWARF (locmap 조인 후) | ✅ | 노드 블록 앞 `#line 23 "model.py"` → 표준 `.debug_line` |

### 4.2 줄 선택 규칙

info_track 이 노드의 `pkg.torch.onnx.stack_trace`(전체 파이썬 콜스택)를 파싱해서,
**`site-packages`/`dist-packages`/`lib/python` 이 아닌 첫 프레임** = 사용자 `forward()` 안의 그 줄을 고른다.
torch 내부 프레임(`pooling.py:224` 등)은 건너뛴다. (`info_track._pick_user_frame`)

### 4.3 세분도 특성

1. **최소 단위 = ONNX 노드** (≈ 레이어/연산자). 한 노드가 lowering 후 op 수십 개로 펼쳐져도 전부 같은 한 줄로 접힌다.
2. **N:1 가능** — 한 Python 줄에서 연산이 여러 개 호출되면 노드도 여러 개, 전부 같은 줄을 가리킨다.
3. **1:다 불가** — 한 노드 안을 문장·피연산자별로 더 잘게 나눌 수 없다.
4. **컬럼 정보 없음** — 파일 + 줄만.

### 4.4 워크된 예시 (`inspect_export_metadata.py` 의 `TinyCNN`)

```python
def forward(self, x):
    x = self.pool(x)              # line 21
    x = x.flatten(1)              # line 22
    x = self.relu(self.fc1(x))    # line 23
    return self.fc2(x)            # line 24
```

| ONNX 노드 | op_type | 매핑되는 원본 줄 |
| --- | --- | --- |
| `node_max_pool2d` | MaxPool | `inspect_export_metadata.py:21` |
| `node_view` | Reshape | `inspect_export_metadata.py:22` |
| `node_linear` | Gemm (fc1) | `inspect_export_metadata.py:23` |
| `node_relu` | Relu | `inspect_export_metadata.py:23` ← 같은 줄 (N:1) |
| `node_linear_1` | Gemm (fc2) | `inspect_export_metadata.py:24` |

→ 5개 노드 / 4개 distinct 줄. `readelf --debug-dump=decodedline` 에 `(inspect_export_metadata.py, {21,22,23,24})` 그대로 존재함.

---

## 5. 트래킹 동작 범위

### 되는 것 (실증 완료)

| 기능 | 동작 | 근거 |
| --- | --- | --- |
| 원본 줄에 중단점 | `gdb break inspect_export_metadata.py:21` → 그 노드 컴파일 코드(`<node_node_max_pool2d+16>`, `0x1159`)에 바인딩 | Part B 4/4 PASS |
| 소스 레벨 스텝 | 한 노드가 op 수십 개로 펼쳐져도 전부 같은 `.py:line` 태그 → `next` 한 번에 노드 하나 통과 | `--EmitMLIR` loc 전파 확인 |
| 역매핑 (PC → 소스) | DWARF `.debug_line` 에 `(model.py, 줄)` 존재 → 크래시 주소/프로파일 샘플을 원본 줄로 | `readelf --debug-dump=decodedline` |
| 재현성 | 같은 입력 3회 바이트 동일 (locmap sha256 `6de7b86…`, 투영 C 동일) | 재현성 체크 |
| 연산자 무관 | MaxPool/Reshape/Gemm/Relu 분기 없이 동일 코드 경로 | `node.name` 기반, op 종류 분기 0 |

### 한계

1. **세분도는 노드까지만.** `x = self.relu(self.fc1(x))` 처럼 한 줄에 두 노드면 `break :23` 이 Gemm·Relu 두 군데 모두에 걸린다(gdb multiple locations). 한 줄 안 중간 지점은 못 잡는다.
2. **변수 추적(값 확인)은 아직 미구현.** `#line` 은 줄 테이블만 만들지 지역변수 위치정보(`DW_AT_location`)를 안 만든다. 생성 C 의 텐서 버퍼는 gdb 에서 C 변수로 보이지만, 원본 `x` / `self.fc1.weight` 같은 PyTorch 이름으로의 매핑 결선은 없다. locmap 에 `fx_node` 문자열이 힌트로만 들어있음.
3. **데모 C 기준.** 실제 harx/EmitC 는 한 노드가 여러 문장으로 펼쳐질 수 있는데, 현재는 노드 시작에 `#line` 한 번 걸고 나머지를 같은 줄로 접는다. 실제 EmitC 포맷 받으면 세분도 재확인 필요.
4. **onnx-mlir CPU 레퍼런스 실행 경로는 이 혜택을 못 받음** — 거긴 `--enable-debug-info` 가 깨져 있고 통과시켜도 `.debug_line` 이 line 0. tail(EmitC→C→gcc) 경로에서만 동작.

---

## 6. 데모 파이프라인 실체 (v3)

### 지금 도커에서 "ONNX→C lowering" 은 실제로 하지 않는다

`code_track.get_generated_c()` 는 **템플릿 C 생성기(대역)** 다. `onnx.load()` 로 그래프를 열어 노드마다
**`(node.name, op_type)` 두 개만** 읽고(provenance 는 안 읽음), 노드당 정확히 7줄짜리 무의미 스텁 함수를 찍는다:

```c
static void node_node_max_pool2d(float *buf, size_t n) {   /* maxpool stand-in */
/* onnx_node_name: node_max_pool2d   op=MaxPool */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* MaxPool */
/* onnx_node_end */
}
```

연산 의미(실제 MaxPool/Gemm 계산)는 없다. 목적은 "노드 정체성을 가진 컴파일 가능한 코드 조각"으로
`#line` 투영을 진짜 DWARF/gdb 로 검증하는 것.

### 컴파일·검증은 실제로 실행한다

이 컨테이너에 없는 건 `harx`/`EmitC`(ONNX→C 부분)이고, `gcc`/`readelf`/`gdb` 는 있다.

| 단계 | 모듈 / 도구 | 입·출력 | 실제 실행? |
| --- | --- | --- | --- |
| 1 | `info_track.build_locmap(onnx)` | ONNX `metadata_props` → locmap dict | ✅ |
| 2 | `code_track.get_generated_c(onnx)` | ONNX `(name, op_type)` → 마커 붙은 데모 C | ✅ (단, **대역**) |
| 3 | `join.inject(c, locmap)` | 마커 뒤에 `#line <줄> "<orig.py>"`, `onnx_node_end` 에서 C 로 되돌림 | ✅ (실제 로직) |
| 4 | `gcc -g -O0` | `.line.c` → 진짜 ELF (`.debug_line` 포함) | ✅ |
| 5 | `readelf --debug-dump=decodedline` + `gdb --batch` | 라인테이블·중단점 바인딩 확인 | ✅ |
| 6 | 1~5 를 3회 반복 | 바이트 동일성 확인 | ✅ |

---

## 7. 검증 결과

### Part A — admit 단계 locmap 캡처 (완료)

- `tinycnn_dynamo.onnx` 5개 노드 전부 위치 매핑 성공 (5/5).
- 각 엔트리: `op_type`, `file`, `line`, `source`, `func`, `module_scope`(예 `fc1.linear`), `namespace`, `class_hierarchy`, `fx_node`, `stack_frames`.
- 조인 키 검증: `--EmitONNXIR` 산출물의 `onnx_node_name` 5개 집합 == locmap 키 5개 집합 (일치).
- 결정성: 같은 `.onnx` → locmap JSON 바이트 동일 (3회 sha256 일치).

### Part A2 — 정식 MLIR→LLVM DWARF 경로(`--enable-debug-info`) 복구 시도 (실패로 기록, 타임박스 준수)

- 실패 pass: stock `DIScopeForLLVMFuncOpPass`.
- 에러: `inlinable function call in a function with a DISubprogram location must have a debug location` (MLIR LLVM verifier). dynamo·mnist 모델 둘 다 같은 지점.
- 원인: 이 패스가 본문 있는 모든 `llvm.func` 에 `DISubprogram` 을 붙이는데, onnx-mlir 이 런타임 헬퍼 호출(`@malloc`, `@omTensor*`, `@printf` …)을 전부 `UnknownLoc` 으로 만들기 때문의 **구조적 문제**.
- 고치려면 새 MLIR 프리패스 + 패스 순서 삽입 + onnx-mlir 전체 재빌드 필요 → CLAUDE.md 빌드 주의 + 타임박스 지침에 따라 **소스 미수정, 진단만 기록**.
- 부차적 이유: 통과시켜도 모델 op 의 `loc` 이 `NameLoc` 이라 `.debug_line` 이 line 0 이 됨. 진짜 줄을 넣으려면 MLIR 단계에서 locmap 을 `FileLineColLoc` 로 주입하는 별도 작업이 필요.

### Part B — tail 단계 `#line` 투영 (완료, readelf/gdb 로 실증) — 4/4 PASS

1. `gcc -g` 가 `.debug_line` 섹션 생성.
2. `.debug_line` 에 원본 `.py` 파일명(`inspect_export_metadata.py`) 등장.
3. locmap 이 투영한 `(inspect_export_metadata.py, {21,22,23,24})` 네 조합이 DWARF 라인테이블에 그대로 존재.
4. `gdb --batch -ex "break inspect_export_metadata.py:21"` → `Breakpoint 1 at 0x1159: file .../inspect_export_metadata.py, line 21`, 그 주소가 `<node_node_max_pool2d+16>`.
   **원본 PyTorch 소스 줄에 건 중단점이 해당 ONNX 노드의 컴파일된 코드 안에 바인딩된다.**

### 재현성 체크 — 통과

| 스크립트 | 반복 | 결과 |
| --- | --- | --- |
| `info_track` (locmap JSON) | 3회 | sha256 동일 |
| `join` `--demo` (`_kernels.c`, `_kernels.line.c`) | 3회 | sha256 동일 |
| `join` 일반 모드 (투영 C) | 3회 | sha256 동일 |

---

## 8. 실제 EmitC/ARX 연동 시 할 일

- [ ] **`code_track.get_generated_c()` 본문 교체** — 입력(경로)·출력(C 문자열) 시그니처 유지. 출력 C 의 노드 구획에 `onnx_node_name` 이 식별 가능하게만 들어있으면 됨 (`join.py` 마커 정규식 `onnx[_-]node(_name)?\s*[:=]\s*<name>` 이 관대하게 인식). `join.py`/`run_full_pipeline.py` 는 안 건드림.
- [ ] 실제 harx/EmitC 산출 C 에 `onnx_node_name` 이 주석 등으로 남는지 확인. 안 남으면 EmitC emitter 쪽에 노드명 주석 추가 협의 (tail 단계라 우리 소유).
- [ ] admit 파이프라인에 `info_track` 을 endpoint 로 결선하고, 산출 locmap 을 tail 까지 전달하는 경로 확정.
- [ ] `#line` 세분도 합의 — 실제 EmitC 는 한 노드가 여러 문장. 노드 시작 1회 `#line` + 이후 접기 방식이 적절한지 디버거 팀과 협의.
- [ ] (선택, 우선순위 낮음) onnx-mlir CPU 실행을 "레퍼런스 실행"으로 쓰려면 `--EmitMLIR` 후 `loc` 에 locmap 을 `FileLineColLoc` 로 심는 별도 패스.
- [ ] 다른 프레임워크(TensorFlow/scikit-learn) ONNX 변환기가 동등한 `metadata_props` 를 채우는지 재검증 (현재는 PyTorch `dynamo=True` 만 검증됨).

---

## 9. 협의자료용 정리 (입력 대상 / 관측 지점 / 출력 형식)

| 구분 | 내용 |
| --- | --- |
| **입력 대상** | `torch.onnx.export(..., dynamo=True)` 산출 ONNX. 노드별 `metadata_props` 에 `pkg.torch.onnx.stack_trace`(파일·줄·소스라인), `namespace`/`class_hierarchy`/`name_scopes`(모듈 계층), `pkg.torch.onnx.fx_node`(FX 표현) 존재. 구식 export 는 신뢰 불가(torch 버전 의존). |
| **관측 지점 1 (admit)** | ONNX 로드 직후. onnx-mlir 이 `metadata_props` 를 버리기 전에 직접 캡처. 조인 키 = `node.name`(= onnx-mlir 의 `onnx_node_name` attr). 출력: `locmap` sidecar JSON. |
| **관측 지점 2 (ONNXIR/MLIR)** | `--EmitONNXIR --mlir-print-debuginfo` 출력의 `loc("<node_name>")` + `{onnx_node_name}` attr. 여기까진 노드 이름만 보존. locmap 과 조인하면 소스 위치 복원. `--EmitMLIR`/`--EmitLLVMIR` 까지 `loc` 이 파생 op 에 전파되므로 이후 단계 조인도 유효. |
| **관측 지점 3 (tail, EmitC→C)** | 생성 C. 각 연산 블록 앞에 `#line <lineno> "<원본.py>"` 주입. 이후 표준 `gcc -g` → DWARF `.debug_line` → gdb 중단점/스텝. `readelf --debug-dump=decodedline` 로 검증. |
| **출력 형식** | (a) locmap sidecar(JSON): `onnx_node_name` → 소스위치·모듈계층. (b) `#line` 이 박힌 C. (c) 표준 DWARF 가 실린 host-mode 실행 아티팩트. |
| **onnx-mlir 정식 DWARF (`--enable-debug-info`)** | 현재 버전에서 미동작(verifier 에러). "모델 코드용 DWARF 는 onnx-mlir 기본 경로로는 안 나온다, `#line` 우회 또는 KrnlInstrument 식 수동 DI 주입이 필요"로 공유. |

### locmap 스키마 (레이어별 정보 측정 항목의 제안 스펙)

| 필드 | 예시 | 의미 / 디버거 용도 |
| --- | --- | --- |
| `op_type` | `Gemm` | ONNX 연산자 종류 |
| `file` | `inspect_export_metadata.py` | 중단점 대상 소스 파일 |
| `line` | `23` | 중단점 대상 줄 |
| `source` | `x = self.relu(self.fc1(x))` | 소스 코드 한 줄(표시용) |
| `func` | `forward` | 소스 함수명 |
| `module_scope` | `fc1.linear` | PyTorch 모듈 경로(트리뷰/네임스코프) |
| `namespace` | `: __main__.TinyCNN/fc1: ...` | torch 원문 계층 문자열 |
| `class_hierarchy` | `["__main__.TinyCNN", "...Linear", "aten.linear.default"]` | 클래스 계층(리스트) |
| `fx_node` | `%linear : ... = call_function[target=torch.ops.aten.linear.default](...)` | 원본 FX 노드 표현(변수 추적 힌트) |
| `stack_frames` | `[{file,line,func,source}, ...]` | 사용자+프레임워크 전체 콜스택 |
| `_meta.join_key` | `onnx_node_name` | onnx-mlir 산출물과의 조인 키 |
| `_meta.onnx_sha256` | `307e6d...` | 어느 ONNX 로 뜬 locmap 인지 고정 |

---

## 10. 산출물 & 재현 명령

**문서**: `debug_info_investigation/debug_info_investigation.md` (1부 조사 / 2부 도구 구현 / 3부 구조 정리)
**산출물**: `debug_info_investigation/artifacts/v3/` (locmap JSON, 데모 C, 투영 C, `pipeline_report.json`, `run_pipeline.log`)

```bash
cd debug_info_investigation

# 전체 파이프라인 (조인키 점검 + build_locmap → get_generated_c → inject → gcc -g → readelf/gdb → 3회 재현성)
./run_pipeline.sh                              # 기본 artifacts/tinycnn_dynamo.onnx
./run_pipeline.sh /path/to/other_dynamo.onnx

# 개별 모듈
python3 info_track.py MODEL.onnx --relative-to /workdir/onnx-mlir -o MODEL.locmap.json
python3 code_track.py MODEL.onnx -o kernels.c
python3 join.py kernels.c MODEL.locmap.json -o kernels.line.c --resolve-paths
python3 run_full_pipeline.py MODEL.onnx --workdir ./artifacts/v3

# onnx-mlir 관측 (참고)
OM=build/Debug/bin/onnx-mlir
$OM --EmitONNXIR --mlir-print-debuginfo --preserveLocations model.onnx -o out
$OM --EmitLLVMIR --mlir-print-debuginfo --preserveLocations model.onnx -o out
$OM --EmitObj    --enable-debug-info    --preserveLocations model.onnx -o out   # ← verifier 에러(재현용)
```
