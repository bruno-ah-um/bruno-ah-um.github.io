---
title: "OpenJDK Babylon Series #3: From Java to Native"
date: 2026-02-04
tags: [ "OpenJDK Babylon", "Java", "MLIR", "TOSA", "code-reflection", "native-compilation"]
description: "Explaining how Java methods can be represented, inspected, and transformed as structured intermediate code models."
cover: mountain.jpg
---

![plug](mountain.jpg)
Foto von <a href="https://unsplash.com/de/@mvds?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mads Schmidt Rasmussen</a> auf <a href="https://unsplash.com/de/fotos/eisbedeckter-berg-bei-tag-xfngap_DToE?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
      
In this post, we demonstrate how Java tensor operations can be compiled automatically to optimized native code. Using Project Babylon's Code Reflection API and MLIR's TOSA dialect, high-level Java methods are transformed into MLIR, lowered to LLVM, and compiled into shared libraries. All without manual boilerplate or marshalling. We cover the full pipeline: Java → Code Model → TOSA MLIR → LLVM → Native Library → Java invocation.

<!--more-->

---

In [Part 1](/posts/babylon_1_mlir_tosa/) we explored MLIR and the TOSA dialect—why TOSA provides a stable, semantically complete target for tensor compilers, and how the MLIR lowering pipeline transforms high-level operations into native code. In [Part 2](/posts/babylon_2_code_relection/) we examined Project Babylon's Code Reflection API—how the `@Reflect` annotation enables Java methods to be captured as structured intermediate representations suitable for analysis and transformation.

Now we bring these concepts together. This post demonstrates how Java tensor operations, written using a fluent API, can be automatically compiled to native x86_64 code through MLIR's TOSA dialect. We'll walk through the complete compilation pipeline and benchmark a real-world example: an MNIST digit classifier that achieves **7.8x speedup** over pure Java execution when compiled through TOSA to native code.

The code is available in the [https://github.com/bruno-ah-um/openjdk-babylon-mlir-tosa](https://github.com/bruno-ah-um/openjdk-babylon-mlir-tosa) under `cr-examples/mlir-tosa`.

---

## Overview of the Compilation Pipeline

The full flow from Java source code to native execution:

```text
┌─────────────────────┐
│  @Reflect Method    │  Java method with tensor operations
└──────────┬──────────┘
           │ Op.ofMethod()
           ▼
┌─────────────────────┐
│  CoreOp.FuncOp      │  Code reflection intermediate representation
└──────────┬──────────┘
           │ TosaCodeGenerator
           ▼
┌─────────────────────┐
│  TOSA MLIR          │  Tensor ops in MLIR's TOSA dialect
└──────────┬──────────┘
           │ mlir-opt (lowering pipeline)
           ▼
┌─────────────────────┐
│  LLVM Dialect       │  Low-level operations for code generation
└──────────┬──────────┘
           │ mlir-translate → clang
           ▼
┌─────────────────────┐
│  Native Library     │  Compiled shared object (.so)
└─────────────────────┘
```

Each stage transforms the representation while preserving semantics, ultimately producing native code that can be invoked directly from Java.

## The Java TOSA API

Before examining the compilation pipeline, let's establish the Java abstractions we're working with.

### Tensor Representation

The `Tensor<T>` class provides a type-safe abstraction over multi-dimensional arrays, backed by Java's Foreign Function & Memory API:

```java
public final class Tensor<T> {

    public enum ElementType {
        FLOAT32(Float.class, ValueLayout.JAVA_FLOAT, 4),
        FLOAT64(Double.class, ValueLayout.JAVA_DOUBLE, 8),
        INT32(Integer.class, ValueLayout.JAVA_INT, 4),
        INT64(Long.class, ValueLayout.JAVA_LONG, 8);
        // ...
    }

    private final long[] shape;
    private final ElementType elementType;
    private final MemorySegment data;  // Off-heap memory
    private final Arena arena;         // Lifecycle management

    // ...
}
```

Using `MemorySegment` for data storage enables zero-copy interoperability with native code—the same memory can be passed directly to compiled MLIR functions without marshalling overhead.

### Creating Tensors

The API provides several factory methods:

```java
// Scalar (0-dimensional)
Tensor<Float> scalar = Tensor.ofScalar(3.14f);

// 1D vector from varargs
Tensor<Float> vector = Tensor.ofFlat(1.0f, 2.0f, 3.0f, 4.0f);

// Multi-dimensional with explicit shape
Tensor<Float> matrix = Tensor.ofShape(new long[]{2, 3},
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f);

// From raw float array (useful for image data)
float[] imageData = loadImage();
Tensor<Float> image = Tensor.ofFloats(new long[]{1, 28, 28, 1}, imageData);
```

### Fluent Operations

The API supports both **static operator calls** and a **fluent chaining style**, allowing developers to choose the approach that best fits their coding style or use case:

```java
// Static API
Tensor<Float> result = TosaOperators.Add(TosaOperators.Mul(a, b), c);

// Fluent API - more readable
Tensor<Float> result = a.mul(b).add(c);

// Complex expressions chain naturally
Tensor<Float> output = input
    .matmul(weights)
    .add(bias)
    .relu();
```

This fluent style maps directly to TOSA operation sequences during code generation.

## A Minimal Example

The simplest function we can compile demonstrates the complete pipeline:

```java
@Reflect
public static Tensor<Float> simpleAdd(Tensor<Float> a, Tensor<Float> b) {
    return a.add(b);
}
```

The `@Reflect` annotation from `jdk.incubator.code` marks this method for code reflection. At runtime, we can obtain its structured representation:

```java
var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);
CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow();
```

The `funcOp` object contains the complete operation tree—parameter declarations, the `add` invocation, and the return statement—in a form suitable for transformation.

## The Compilation Pipeline

### Stage 1: Code Model to TOSA MLIR

The `TosaCodeGenerator` traverses the code model and emits corresponding TOSA operations. For our `simpleAdd` function:

```java
String tosaMlir = TosaCodeGenerator.generateTosa(funcOp, "simpleAdd", tensorRank);
```

This produces:

```mlir
module {
  func.func @simpleAdd(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}
```

The generator walks the code model's operation tree:

```java
private static MemorySegment processOp(Op op, GeneratorContext ctx, MemorySegment function) {
    return switch (op) {
        case CoreOp.VarOp varOp -> {
            // Variable declarations - map the var to the input value
            Value inputValue = varOp.operands().getFirst();
            MemorySegment handle = ctx.valueHandles.get(inputValue);
            if (handle != null) {
                ctx.varToHandle.put(varOp.result(), handle);
            }
            yield MemorySegment.NULL;
        }
        case JavaOp.InvokeOp invokeOp -> {
            yield processInvokeOp(invokeOp, ctx, function);
        }
        case CoreOp.ReturnOp returnOp -> {
            Value returnValue = returnOp.operands().getFirst();
            MemorySegment valueHandle = ctx.valueHandles.get(returnValue);
            // ... emit return
        }
        default -> MemorySegment.NULL;
    };
}
```

When encountering method invocations like `add`, `mul`, or `matmul`, the generator maps them to native TOSA operations via the MLIR C API:

```java
case "Add", "add" -> {
    if (operandHandles.size() >= 2) {
        yield mlir_tosa_c_api_h.mlir_tosa_add(function,
            operandHandles.get(0), operandHandles.get(1), resultType);
    }
    yield MemorySegment.NULL;
}
```

### Stage 2: TOSA Lowering Pipeline

The generated TOSA MLIR must be lowered through several dialect conversions before reaching LLVM. The `TosaCompiler` invokes `mlir-opt` with a comprehensive pass pipeline:

```java
command.add("--pass-pipeline=builtin.module(" +
    "func.func(tosa-to-linalg-named)," +  // Named ops (matmul, conv2d)
    "func.func(tosa-to-linalg)," +         // Element-wise ops
    "func.func(tosa-to-arith)," +          // Constants
    "func.func(tosa-to-tensor)," +         // Tensor operations
    "func.func(linalg-fuse-elementwise-ops)," +  // Optimization
    "one-shot-bufferize{bufferize-function-boundaries}," +  // Tensor → MemRef
    "func.func(convert-linalg-to-loops)," +  // Generate loop nests
    "func.func(lower-affine)," +
    "func.func(arith-expand)," +
    "func.func(convert-scf-to-cf)," +
    "convert-arith-to-llvm," +
    "convert-func-to-llvm," +
    "convert-cf-to-llvm," +
    "finalize-memref-to-llvm," +
    "reconcile-unrealized-casts" +
    ")");
```

This pipeline performs several key transformations:

1. **TOSA to Linalg**: Converts tensor operations to Linalg's structured operation form
2. **Bufferization**: Transforms immutable tensors to mutable memory references (memrefs)
3. **Loop generation**: Expands Linalg operations into explicit loop nests
4. **LLVM conversion**: Lowers all operations to LLVM dialect

After this pipeline, our simple add becomes explicit loops over memory. Here's a condensed view of the key sections:

```mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Each tensor is unpacked into: allocated_ptr, aligned_ptr, offset, size[0], stride[0]
  llvm.func @simpleAdd(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64,
                       %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64)
      -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {

    // Build memref descriptor structs from unpacked parameters
    %lhs = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %lhs1 = llvm.insertvalue %arg0, %lhs[0] : ...  // allocated ptr
    %lhs2 = llvm.insertvalue %arg1, %lhs1[1] : ... // aligned ptr
    // ... offset, size, stride

    // Allocate result buffer via malloc with 64-byte alignment
    %size = llvm.extractvalue %lhs_final[3, 0] : ...  // get dimension size
    %bytes = llvm.getelementptr %null[%size] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %alloc = llvm.call @malloc(%bytes_plus_alignment) : (i64) -> !llvm.ptr
    // ... align to 64-byte boundary

    // Build result memref descriptor
    %result = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %result1 = llvm.insertvalue %alloc, %result[0] : ...
    // ...

    // Main computation loop
    llvm.br ^loop_header(%zero : i64)
  ^loop_header(%i: i64):
    %cond = llvm.icmp "slt" %i, %size : i64
    llvm.cond_br %cond, ^loop_body, ^loop_exit
  ^loop_body:
    // Load elements from both input tensors
    %lhs_ptr = llvm.extractvalue %lhs_final[1] : ...
    %lhs_elem_ptr = llvm.getelementptr %lhs_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lhs_val = llvm.load %lhs_elem_ptr : !llvm.ptr -> f32

    %rhs_ptr = llvm.extractvalue %rhs_final[1] : ...
    %rhs_elem_ptr = llvm.getelementptr %rhs_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %rhs_val = llvm.load %rhs_elem_ptr : !llvm.ptr -> f32

    // The actual addition!
    %sum = llvm.fadd %lhs_val, %rhs_val : f32

    // Store to result
    %res_ptr = llvm.extractvalue %result_final[1] : ...
    %res_elem_ptr = llvm.getelementptr %res_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %sum, %res_elem_ptr : f32, !llvm.ptr

    %next = llvm.add %i, %one : i64
    llvm.br ^loop_header(%next : i64)
  ^loop_exit:
    llvm.return %result_final : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}
```

The actual output is ~200 lines due to broadcast handling and the verbosity of SSA form, but the essential structure is: unpack parameters into structs, allocate output, loop over elements, perform the operation, return the result descriptor.

### Stage 3: LLVM IR Generation

The LLVM dialect is translated to standard LLVM IR:

```bash
mlir-translate --mlir-to-llvmir func_llvm.mlir -o func.ll
```

### Stage 4: Native Compilation

Finally, Clang compiles the LLVM IR to a shared library:

```bash
clang -shared -fPIC -O2 -o libsimpleAdd.so func.ll
```

The `-O2` flag enables LLVM's optimization passes, producing efficient native code.

## Native Invocation

With the shared library compiled, we need to invoke it from Java. The `CompiledFunction` class handles the FFM integration.

### Loading the Library

```java
Arena arena = Arena.ofAuto();
SymbolLookup lookup = SymbolLookup.libraryLookup(soFile, arena);

MemorySegment funcAddr = lookup.find(funcName).orElseThrow();
```

### The MemRef ABI

MLIR's lowering produces functions that use the **memref calling convention**. Each tensor parameter becomes multiple values:

```
{allocated_ptr, aligned_ptr, offset, sizes[rank], strides[rank]}
```

For a 1D tensor, this means 5 values: two pointers, an offset, one size, and one stride. The function descriptor must reflect this:

```java
private FunctionDescriptor buildFunctionDescriptor(int numParams, int rank) {
    List<MemoryLayout> paramLayouts = new ArrayList<>();

    for (int i = 0; i < numParams; i++) {
        paramLayouts.add(ValueLayout.ADDRESS);   // allocated ptr
        paramLayouts.add(ValueLayout.ADDRESS);   // aligned ptr
        paramLayouts.add(ValueLayout.JAVA_LONG); // offset

        for (int d = 0; d < rank; d++) {
            paramLayouts.add(ValueLayout.JAVA_LONG); // size[d]
        }
        for (int d = 0; d < rank; d++) {
            paramLayouts.add(ValueLayout.JAVA_LONG); // stride[d]
        }
    }

    // Return type is also a struct with the same layout
    MemoryLayout returnLayout = MemoryLayout.structLayout(/* ... */);

    return FunctionDescriptor.of(returnLayout, paramLayouts.toArray(new MemoryLayout[0]));
}
```

### Marshalling Tensors

The `invoke` method marshals Java `Tensor` objects to the native ABI:

```java
public <T> Tensor<T> invoke(Tensor<?>... tensors) {
    try (Arena arena = Arena.ofConfined()) {
        Object[] args = new Object[1 + totalValues];

        // First arg: SegmentAllocator for return struct
        args[0] = (SegmentAllocator) arena;

        int argIndex = 1;
        for (Tensor<?> tensor : tensors) {
            MemorySegment data = tensor.data();
            long[] shape = tensor.shape();

            args[argIndex++] = data;  // allocated ptr
            args[argIndex++] = data;  // aligned ptr
            args[argIndex++] = 0L;    // offset

            // Sizes
            for (int d = 0; d < rank; d++) {
                args[argIndex++] = shape[d];
            }

            // Strides (row-major)
            for (int d = 0; d < rank; d++) {
                long stride = 1;
                for (int k = d + 1; k < rank; k++) {
                    stride *= shape[k];
                }
                args[argIndex++] = stride;
            }
        }

        // Invoke native function
        MemorySegment resultStruct = (MemorySegment) handle.invokeWithArguments(args);

        // Extract result tensor from return struct
        // ...
    }
}
```

The `Tensor.data()` method returns a `MemorySegment` pointing to off-heap memory, which is passed directly to the native function—no copying required.

## Putting It Together

Here's the complete compilation and execution flow:

```java
// 1. Create the compiler
TosaCompiler compiler = new TosaCompiler(true);  // verbose mode

// 2. Get the method to compile
var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);

// 3. Compile to native code
CompiledFunction fn = compiler.compile(method);

// 4. Create input tensors
Tensor<Float> a = Tensor.ofShape(new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
Tensor<Float> b = Tensor.ofShape(new long[]{4}, 10.0f, 20.0f, 30.0f, 40.0f);

// 5. Invoke native function
Tensor<Float> result = fn.invoke(a, b);

System.out.println(result);
// Tensor{shape=[4], type=FLOAT32, data=[11.0, 22.0, 33.0, 44.0]}
```

With verbose mode enabled, the compiler prints each stage:

```
[TosaCompiler] Generating TOSA MLIR (rank=1)...
[TosaCompiler] TOSA MLIR:
module {
  func.func @simpleAdd(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}
[TosaCompiler] Lowering TOSA -> LLVM dialect...
[TosaCompiler] Translating to LLVM IR...
[TosaCompiler] Compiling to shared library...
[TosaCompiler] Loading shared library: /tmp/tosa_compile_xxx/libsimpleAdd.so
```

## Beyond Simple Addition

The same pipeline handles complex compositions. Consider a two-layer neural network:

```java
@Reflect
public static Tensor<Float> twoLayerMLP(Tensor<Float> x,
                                        Tensor<Float> w1, Tensor<Float> b1,
                                        Tensor<Float> w2, Tensor<Float> b2) {
    // Layer 1: hidden = ReLU(x @ w1 + b1)
    Tensor<Float> hidden = x.matmul(w1).add(b1).relu();
    // Layer 2: output = hidden @ w2 + b2
    return hidden.matmul(w2).add(b2);
}
```

This generates TOSA MLIR with `tosa.matmul`, `tosa.add`, and `tosa.clamp` (for ReLU) operations, all lowered and compiled together. The native library performs the complete forward pass in optimized code.

For convolutional networks, operations like `Conv2D` and `MaxPool2D` follow the same pattern:

```java
@Reflect
public static Tensor<Float> convBlock(Tensor<Float> input,
                                      Tensor<Float> weight,
                                      Tensor<Float> bias) {
    return TosaOperators.Conv2D(input, weight, bias,
        new long[]{0, 0, 0, 0},  // padding
        new long[]{1, 1},         // stride
        new long[]{1, 1})         // dilation
        .relu();
}
```

## MNIST Benchmark: Measuring the Speedup

To quantify the performance benefit of native compilation, we benchmark a complete MNIST digit classifier—a classic machine learning problem where the goal is to recognize handwritten digits (0-9) from 28x28 grayscale images.

### The Model Architecture

We implement a LeNet-style convolutional neural network, a proven architecture for MNIST that includes both compute-intensive convolutions and memory-bound fully connected layers:

```java
@Reflect
public static Tensor<Float> mnistModel(
        Tensor<Float> input,                              // [1, 28, 28, 1]
        Tensor<Float> conv1Weight, Tensor<Float> conv1Bias,  // Conv 5x5, 6 filters
        Tensor<Float> conv2Weight, Tensor<Float> conv2Bias,  // Conv 5x5, 16 filters
        Tensor<Float> fc1Weight, Tensor<Float> fc1Bias,      // 256 -> 120
        Tensor<Float> fc2Weight, Tensor<Float> fc2Bias,      // 120 -> 84
        Tensor<Float> fc3Weight, Tensor<Float> fc3Bias) {    // 84 -> 10

    // Conv block 1: [1,28,28,1] -> [1,12,12,6]
    Tensor<Float> conv1 = TosaOperators.Conv2D(input, conv1Weight, conv1Bias,
        new long[]{0,0,0,0}, new long[]{1,1}, new long[]{1,1});
    Tensor<Float> pool1 = TosaOperators.MaxPool2D(conv1.relu(),
        new long[]{2,2}, new long[]{2,2}, new long[]{0,0,0,0});

    // Conv block 2: [1,12,12,6] -> [1,4,4,16]
    Tensor<Float> conv2 = TosaOperators.Conv2D(pool1, conv2Weight, conv2Bias,
        new long[]{0,0,0,0}, new long[]{1,1}, new long[]{1,1});
    Tensor<Float> pool2 = TosaOperators.MaxPool2D(conv2.relu(),
        new long[]{2,2}, new long[]{2,2}, new long[]{0,0,0,0});

    // Flatten and FC layers: [1,4,4,16] -> [1,1,256] -> ... -> [1,1,10]
    Tensor<Float> flat = TosaOperators.Reshape(pool2, new long[]{1, 1, 256});
    Tensor<Float> fc1 = flat.matmul(fc1Weight).add(fc1Bias).relu();
    Tensor<Float> fc2 = fc1.matmul(fc2Weight).add(fc2Bias).relu();
    return fc2.matmul(fc3Weight).add(fc3Bias);
}
```

This model has approximately 44,000 parameters and performs the complete inference pipeline: convolutions, activations, pooling, and dense layers.

### Benchmark Results

We compare three configurations:
- **Java**: Pure Java execution using the Tensor API
- **Native**: MLIR-compiled native code via the TOSA pipeline

After 500 warmup iterations and 2,000 measurement iterations:

```
======================================================================
SUMMARY: MNIST Complete Model Benchmark
======================================================================

| Layer               | Java (μs)   | Native (μs)  | Speedup  |
|---------------------|-------------|--------------|----------|
| Conv Layers         |     1842.35 |       194.67 |    9.46x |
| FC Layers           |      156.28 |        52.14 |    3.00x |
| Full Model (e2e)    |     2036.71 |       259.83 |    7.84x |

Throughput (images/sec):
  Java:   491 images/sec
  Native: 3,849 images/sec

Native TOSA compilation provides 7.8x speedup for complete MNIST inference!
```

The convolutional layers show the largest speedup (~9.5x) because they benefit most from LLVM's loop optimizations and efficient memory access patterns. The fully connected layers, being more memory-bound, still achieve a 3x improvement. End-to-end, the native-compiled model processes **nearly 8 times more images per second** than the pure Java implementation.

### Why Native is Faster

Several factors contribute to the performance difference:

1. **Loop optimization**: LLVM applies loop unrolling, vectorization hints, and instruction scheduling
2. **Memory layout**: The compiled code uses cache-friendly access patterns generated by the bufferization pass
3. **Reduced allocation**: The MLIR pipeline can fuse operations and minimize intermediate allocations

The Java implementation, while benefiting from JIT compilation, still incurs per-operation dispatch overhead and less aggressive loop transformations.

## Summary

This post demonstrated the complete integration between Project Babylon's code reflection and MLIR's TOSA dialect:

1. **Code Reflection** captures Java methods as analyzable intermediate representations
2. **TosaCodeGenerator** transforms the code model to TOSA MLIR operations
3. **MLIR's lowering pipeline** compiles TOSA through Linalg to LLVM dialect
4. **Native compilation** produces optimized shared libraries
5. **FFM integration** enables zero-copy invocation from Java

The MNIST benchmark demonstrates the practical impact: a complete CNN achieves **7.8x speedup** when compiled through this pipeline, processing nearly 4,000 images per second compared to ~500 with pure Java. The result: tensor operations written in idiomatic Java, automatically compiled to native code through a production-quality compiler infrastructure.

## What's Next

Future work includes:
- GPU execution via MLIR's GPU dialects
- Automatic differentiation for training workloads
- Integration with the broader Babylon ecosystem for domain-specific optimizations

The code is available in the [https://github.com/bruno-ah-um/openjdk-babylon-mlir-tosa](https://github.com/bruno-ah-um/openjdk-babylon-mlir-tosa) under `cr-examples/mlir-tosa`.
