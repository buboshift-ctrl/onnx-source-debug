# ONNX Source Debug

Enable source-level debugging of ONNX models by preserving PyTorch source locations through `#line` directives, compatible with standard GCC/DWARF/GDB tools.

## Overview

**Problem**: ONNX models lack source location information from their original PyTorch code, making it impossible to set breakpoints or trace execution in the original Python source.

**Solution**: This project preserves source location information from PyTorch's ONNX export metadata (`metadata_props`) and injects it into generated C code via `#line` directives, enabling standard GDB debugging of original Python source lines.

## Architecture

```
PyTorch Model
    ↓
torch.onnx.export(dynamo=True)
    ↓
ONNX File (with metadata_props)
    ↓
info_track.py      → Extract source locations → locmap.json
    ↓
code_track.py      → Generate C code → kernels.c
    ↓
join.py            → Inject #line directives → kernels.line.c
    ↓
gcc -g             → Binary + DWARF
    ↓
gdb                → Debug original Python source!
```

## Quick Start

### Installation

```bash
git clone https://github.com/Zekrom0644/onnx-source-debug.git
cd onnx-source-debug
pip install -r requirements.txt
```

### Usage

```bash
cd debug_info_investigation
python3 run_full_pipeline.py /path/to/model.onnx
```

Output:
- `model.locmap.json` - Source location mappings
- `model_kernels.line.c` - C code with #line directives
- `model_kernels_dbg` - Debuggable binary

### Debug with GDB

```bash
gdb ./artifacts/v3/tinycnn_dynamo/tinycnn_dynamo_kernels_dbg
(gdb) break inspect_export_metadata.py:23
(gdb) run
(gdb) bt
```

## Evaluation Results

3 models, 154 nodes:
- ✅ 100% location preservation
- ✅ 100% join success
- ✅ DWARF/GDB verification passed
- ✅ ~25% code overhead

See [debug_info_investigation/paper_eval/RESULTS.md](debug_info_investigation/paper_eval/RESULTS.md)

## Requirements

- Python 3.8+
- ONNX 1.14+
- GCC, GDB

## License

MIT
