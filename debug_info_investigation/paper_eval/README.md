# Paper Evaluation Results

Comprehensive evaluation of ONNX source debug pipeline across 3 models.

## Contents

- **RESULTS.md** - Main evaluation report with tables and findings
- **eval_metrics.py** - Script to extract basic metrics
- **detailed_analysis.py** - Advanced analysis (Conv-BN fusion, frame policies)
- **final_report.py** - Report generation from analysis

## Summary Statistics

| Metric | Value |
|--------|-------|
| Models Tested | 3 (TinyCNN, ResNet18, MobileNetV2) |
| Total Nodes | 154 |
| Location Preservation | 100% |
| Join Success | 100% (154/154) |
| DWARF Verification | ✓ All pass |
| GDB Breakpoint Binding | ✓ All pass |
| Code Overhead | ~25% |

## Test Models

### TinyCNN (5 nodes)
- Simple CNN: Conv → MaxPool → FC → ReLU → FC
- All operations in single file
- No fusion effects
- ✓ 5/5 nodes mapped

### ResNet18 (49 nodes)
- Pre-trained from torchvision
- Conv-BatchNorm fusion: 20 Conv nodes → batchnorm.py
- ✓ 49/49 nodes mapped
- Demonstrates fusion limitations

### MobileNetV2 (100 nodes)
- Pre-trained from torchvision  
- Conv-BatchNorm fusion: 52 Conv nodes → batchnorm.py
- Multiple Clip operations
- ✓ 100/100 nodes mapped
- Largest test case

## Key Findings

1. **Perfect Location Preservation**: 100% of nodes have source location info
2. **DWARF Integration Works**: #line directives correctly map to DWARF
3. **GDB Support**: Original Python source breakpoints function properly
4. **Conv-BN Fusion**: torchvision models show automatic fusion effects
5. **Minimal Overhead**: ~25% additional lines from #line directives

## Evaluation Scripts

### eval_metrics.py
Extracts basic metrics from each model:
```bash
python3 eval_metrics.py
```

Output: Node counts, location rates, op_type distribution

### detailed_analysis.py
Analyzes Conv-BN fusion and frame selection policies:
```bash
python3 detailed_analysis.py
```

Output: Fusion patterns, policy comparison

### final_report.py
Generates complete evaluation report:
```bash
python3 final_report.py
```

Output: RESULTS.md with all tables and findings

## How to Reproduce

1. Ensure test models exist in `../artifacts/v3/`
2. Run pipeline to generate locmap and pipeline reports:
```bash
cd ..
python3 run_full_pipeline.py artifacts/tinycnn_dynamo.onnx
```

3. Generate evaluation report:
```bash
python3 paper_eval/final_report.py
```

## For Publication

Key statements from RESULTS.md:

> "We preserve ONNX source location information from PyTorch export metadata
> by injecting #line directives into generated C code, enabling standard GDB/DWARF 
> debugging of original Python source lines. 100% location preservation across 
> 154 nodes; 25% code overhead; all DWARF verification checks pass."

## Limitations

- Conv-BatchNorm fusion maps to fusion point (not original Conv source)
- Only validated with PyTorch dynamo exporter
- Requires torch.onnx.export(..., dynamo=True)
- Code overhead from #line directives

## Future Work

- Support other frameworks (TensorFlow, JAX)
- Optimize frame selection for fused operations
- Integration with actual onnx-mlir EmitC
- Binary size optimization for DWARF
