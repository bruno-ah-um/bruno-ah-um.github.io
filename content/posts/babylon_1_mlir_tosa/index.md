---
title: "OpenJDK Babylon Series: #1 From High-Level Tensor Operations to MLIR TOSA"
date: 2026-01-14
tags: [ "Java", "OpenJDK Babylon", "MLIR", "TOSA"]
description: "targeting MLIR TOSA enables efficient lowering of Babylon’s Java tensor code to native hardware."
cover: mlir.png
---

![mlir](mlir.png)

[Project Babylon](https://openjdk.org/projects/babylon/) extends Java beyond the JVM, enabling compiler-style transformation of Java code into alternative execution models such as machine learning runtimes and hardware accelerators through its Code Reflection infrastructure.

[Babylon can export Java programs to ONNX](https://github.com/openjdk/babylon/tree/code-reflection/cr-examples/onnx), but if the real goal is efficient code generation for CPUs, GPUs, NPUs, and custom accelerators, ONNX is not always the best target. This post argues for MLIR [TOSA](https://www.mlplatform.org/tosa) as a more suitable compiler IR—one designed to lower cleanly across architectures and backend targets.

<!--more-->

---

## Introducing LLVM MLIR

To understand why TOSA is a strong target, we first need LLVM MLIR. MLIR (Multi-Level Intermediate Representation) is a compiler framework designed for domain-specific languages and accelerator-oriented compilation. It generalizes LLVM IR by supporting custom operations, types, and abstractions.

Programs in MLIR are represented in SSA form using operations organized into blocks and regions, enabling structured and hierarchical IRs. This makes it possible to model everything from high-level tensor programs to low-level hardware constructs.

MLIR’s defining feature is its dialect system: developers can introduce new operations and types as self-contained dialects, loaded without recompiling the compiler. This makes MLIR especially well-suited for rapidly evolving domains like machine learning and hardware accelerators.

---

## What Is TOSA?

[TOSA (Tensor Operator Set Architecture)](https://www.mlplatform.org/tosa) is a first-class MLIR dialect that defines a compact, stable set of tensor operations for machine learning compilers and accelerator backends.

It is built around a few core ideas: a minimal but expressive operator set, precisely defined semantics, no framework- or training-specific behavior, and mechanical lowering to lower-level IRs.

TOSA covers common tensor workloads—element-wise ops, reductions, data movement, and neural network primitives such as conv2d and matmul. Because every operation has fully specified semantics, lowering to lower-level MLIR dialects or hardware-specific code is predictable and unambiguous.

---

## TOSA vs. ONNX: A Compiler-Centric Comparison

One practical but often overlooked difference is **where these IRs live**:

> **TOSA is a first-class dialect inside LLVM MLIR. ONNX is not.**

This distinction has deep consequences for compiler architecture, maintenance, and long-term stability.

| Aspect           | TOSA                         | ONNX                                        |
| ---------------- | ---------------------------- | ------------------------------------------- |
| Primary goal     | Compiler IR                  | Model interchange                           |
| MLIR integration | **First-class MLIR dialect** | External format, requires import/conversion |
| Operator set     | ~80 focused ops              | 180+ ops, overlapping                       |
| Semantics        | Fully specified              | Often framework-defined                     |
| Versioning       | Stable spec                  | Frequent opset changes                      |
| Lowering path    | Direct to Linalg             | Often via TOSA anyway                       |

The key insight is this:

> **When you target TOSA, you are already speaking MLIR.**
> There is no foreign IR to import, normalize, or continuously chase as specifications evolve.

> **Many MLIR-based compilers lower ONNX *into TOSA* before doing real work.**

If your frontend already exposes structured tensor semantics—whether from Java, DSLs, or domain-specific IRs—targeting TOSA directly removes an entire translation layer and keeps the compiler stack **inside MLIR from day one**.

---

## Generating TOSA Programmatically

TOSA is not just a textual format—it is a first-class MLIR dialect with a full C++ API. This makes it ideal as a compilation target from custom frontends.

Below is a minimal example that generates a vector addition function in TOSA using MLIR’s C++ builder API:

```cpp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

int main() {
    // Initialize context and load dialects
    MLIRContext context;
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<tosa::TosaDialect>();

    // Create module and builder
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());

    // Define tensor type: tensor<?xf32>
    auto tensorType = RankedTensorType::get({ShapedType::kDynamic}, builder.getF32Type());

    // Create function: (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    auto funcType = builder.getFunctionType({tensorType, tensorType}, {tensorType});
    auto funcOp = builder.create<func::FuncOp>(loc, "vector_add", funcType);
    funcOp.setSymVisibility("public");

    // Build function body
    Block *entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto addOp = builder.create<tosa::AddOp>(loc, tensorType,
                                              entry->getArgument(0),
                                              entry->getArgument(1));
    builder.create<func::ReturnOp>(loc, addOp.getResult());

    // Verify and print
    if (failed(verify(module)))
        return 1;
    module.dump();
    return 0;
}
```

Which produces:

```mlir
module {
  func.func @vector_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}
```

---

## The Lowering Pipeline: TOSA to Native Code

Once we have TOSA IR, we need to lower it through several abstraction levels to reach native machine code. MLIR provides a well-defined path:

```
TOSA (tensor operations)
    ↓
Linalg (structured linear algebra)
    ↓
Bufferization (tensors → memrefs)
    ↓
SCF (structured control flow / loops)
    ↓
LLVM Dialect
    ↓
LLVM IR → Native Code
```

Each stage reduces abstraction while preserving semantics:

| Stage | Transformation |
|-------|----------------|
| **TOSA → Linalg** | High-level tensor ops become explicit element-wise computations |
| **Bufferization** | Immutable tensors become mutable memory buffers (`memref`) |
| **Linalg → SCF** | Structured ops become `scf.for` loops |
| **SCF → CF** | Structured loops become branches and jumps |
| **→ LLVM** | Everything converts to LLVM dialect primitives |

The entire transformation can be executed with a single `mlir-opt` invocation using `--pass-pipeline`:

```bash
mlir-opt input.mlir --pass-pipeline='builtin.module(
    func.func(tosa-make-broadcastable, tosa-infer-shapes, tosa-to-linalg, tosa-to-arith),
    one-shot-bufferize{bufferize-function-boundaries},
    buffer-deallocation-pipeline,
    convert-linalg-to-loops,
    lower-affine,
    convert-scf-to-cf,
    convert-arith-to-llvm,
    convert-math-to-llvm,
    convert-index-to-llvm,
    finalize-memref-to-llvm,
    convert-func-to-llvm,
    reconcile-unrealized-casts,
    canonicalize
)' -o output_llvm.mlir
```

From there, standard LLVM tools take over:

```bash
mlir-translate output_llvm.mlir --mlir-to-llvmir -o output.ll
llc output.ll -march=x86-64 -filetype=obj -relocation-model=pic -o output.o
gcc -shared output.o -o libvector_add.so
```

The result is a native shared library that can be called directly from C.

### The Memref Calling Convention

MLIR uses a **memref descriptor** ABI for tensor arguments. Each 1D tensor becomes 5 parameters:

| Field | Purpose |
|-------|---------|
| `allocated` | Base pointer (for deallocation) |
| `aligned` | Aligned data pointer (for access) |
| `offset` | Offset into the buffer |
| `size` | Number of elements |
| `stride` | Stride between elements |

The raw function signature looks like:

```c
typedef struct {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
} MemRef1D;

extern MemRef1D vector_add(
    float *a_alloc, float *a_aligned, int64_t a_offset, int64_t a_size, int64_t a_stride,
    float *b_alloc, float *b_aligned, int64_t b_offset, int64_t b_size, int64_t b_stride
);
```

For simple contiguous arrays, a wrapper function can hide this complexity:

```c
void vector_add_simple(float *a, float *b, float *out, int64_t size) {
    MemRef1D result = vector_add(a, a, 0, size, 1, b, b, 0, size, 1);
    for (int64_t i = 0; i < size; i++) {
        out[i] = result.aligned[i];
    }
    free(result.allocated);
}

// Clean API for users
vector_add_simple(a, b, result, size);
```

This calling convention may seem verbose, but it enables MLIR to support strided tensors, views, and slices without copying data—essential for efficient tensor operations.

---


