"""MLIR Lowering Passes for Kairo

This package contains lowering passes that transform Kairo's high-level
dialects into progressively lower-level representations:

Kairo Dialects → SCF/Arith/Func → LLVM Dialect → LLVM IR → Native Code

Passes:
- FieldToSCFPass: Lower field operations to structured control flow
- SCFToLLVMPass: Lower SCF to LLVM dialect
- OptimizationPasses: MLIR optimization passes
"""

# TODO: Import passes as implemented
# from .field_to_scf import FieldToSCFPass
# from .scf_to_llvm import SCFToLLVMPass
# from .optimization import create_optimization_pipeline

__all__ = []
