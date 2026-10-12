# Debug Info Investigation

Core modules for ONNX source location preservation.

## Modules

### 1. info_track.py (277 lines)

**Purpose**: Extract source location information from ONNX metadata

**Input**: ONNX file (from `torch.onnx.export(..., dynamo=True)`)

**Output**: `locmap.json` mapping node names to source locations

**Key Functions**:
- `build_locmap(onnx_path)` → dict
- `dumps_deterministic(locmap)` → str (byte-identical output)

**Process**:
1. Read ONNX node's `metadata_props` field
2. Extract `pkg.torch.onnx.stack_trace` 
3. Parse Python stack trace with regex
4. Filter PyTorch framework frames (site-packages, lib/python)
5. Select innermost user code frame
6. Structure node info → JSON

**Example Output**:
```json
{
  "nodes": {
    "node_conv_0": {
      "op_type": "Conv",
      "file": "model.py",
      "line": 23,
      "source": "x = self.conv(x)",
      "func": "forward",
      "module_scope": "layer1.conv",
      "stack_frames": [...]
    }
  }
}
```

### 2. code_track.py (131 lines)

**Purpose**: Generate C code from ONNX graph (demo implementation)

⚠️ **Note**: This is a stand-in. In production, use actual `onnx-mlir` EmitC output.

**Input**: ONNX file

**Output**: `kernels.c` with one function per node

**Key Functions**:
- `get_generated_c(onnx_path)` → str

**Example Output**:
```c
/* onnx_node_name: node_conv_0 */
void node_node_conv_0() {
  // Conv operation code
  float *input = ...;
  float *output = malloc(...);
  // ... simple operation
}

/* onnx_node_name: node_relu_0 */
void node_node_relu_0() { ... }
```

**Features**:
- Independent of locmap (doesn't read source info)
- Uses only node.name and op_type
- Deterministic output
- Easy to replace with real EmitC

### 3. join.py (187 lines)

**Purpose**: Inject `#line` directives into C code

**Input**: 
- `kernels.c` (from code_track)
- `locmap.json` (from info_track)

**Output**: `kernels.line.c` with source location directives

**Key Functions**:
- `inject(c_source, locmap)` → str
- `inject_with_stats(...)` → (str, dict)
- `find_node_markers(c_source)` → list[str]

**Join Key**: `node.name` (ONNX node name)

**Example Output**:
```c
/* onnx_node_name: node_conv_0 */
#line 23 "model.py"
void node_node_conv_0() {
  // Original C code
}
#line 45 "kernels.line.c"  // Reset to actual C file
```

**Features**:
- Pure text transformation
- Deterministic
- No dependencies on info_track or code_track internals
- Standard GCC/DWARF compatible

### 4. run_full_pipeline.py (234 lines)

**Purpose**: Orchestrate full pipeline + verification

**Input**: ONNX file

**Output**:
- `locmap.json`
- `kernels.c`
- `kernels.line.c`
- `kernels_dbg` (compiled binary)
- `pipeline_report.json` (verification results)

**Pipeline**:
1. info_track.build_locmap()
2. code_track.get_generated_c()
3. join.inject()
4. gcc -g (compile with debug info)
5. readelf/gdb (verify DWARF)

**Key Functions**:
- `run(onnx_path, workdir)` → dict
- `verify_dwarf(binary, locmap)` → dict
- `reproducibility_check(onnx_path)` → dict

**Verification Checks**:
- ✓ DWARF .debug_line section created
- ✓ Original .py files appear in DWARF
- ✓ Line projection accurate
- ✓ GDB breakpoint binding works

## directories

### artifacts/

Test results from running the pipeline on 3 models:

```
v3/tinycnn_dynamo/          (5 nodes)
  ├─ tinycnn_dynamo.onnx
  ├─ tinycnn_dynamo.locmap.json
  ├─ tinycnn_dynamo_kernels.c
  ├─ tinycnn_dynamo_kernels.line.c
  ├─ tinycnn_dynamo_kernels_dbg (binary + DWARF)
  └─ tinycnn_dynamo_pipeline_report.json

v3/resnet18_dynamo/         (49 nodes)
v3/mobilenet_v2_dynamo/     (100 nodes)
```

### paper_eval/

Evaluation results and analysis:

- `RESULTS.md` - Comprehensive evaluation report
- `eval_metrics.py` - Basic metric extraction
- `detailed_analysis.py` - Conv-BN fusion analysis
- `final_report.py` - Report generation

## Usage

Run on a single model:

```bash
python3 run_full_pipeline.py /path/to/model.onnx
```

Run on all test models:

```bash
bash run_pipeline.sh
```

## Key Characteristics

All modules are:
- **Deterministic**: Same input → identical output
- **Independent**: info_track, code_track, join don't depend on each other
- **Modular**: Easy to replace individual components
- **Simple**: No complex dependencies

## For Production

To use with real `onnx-mlir` EmitC:

1. Replace `code_track.get_generated_c()` with actual EmitC output
2. Keep all other modules unchanged
3. Signature must stay: `get_generated_c(onnx_path: str) -> str`

## Testing

Verify reproducibility (3 iterations, byte-identical):

```bash
python3 run_full_pipeline.py model.onnx --reproducibility-check
```

Verify DWARF/GDB integration:

```bash
# Check DWARF line table
readelf --debug-dump=decodedline model_kernels_dbg | grep ".py"

# Test GDB breakpoint
gdb --batch -ex "break model.py:23" model_kernels_dbg
```
