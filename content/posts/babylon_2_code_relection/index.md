---
title: "OpenJDK Babylon Series: #2 Code Reflection in Java"
date: 2026-01-21
tags: ["OpenJDK Babylon"]
description: "Explaining how Java methods can be represented, inspected, and transformed as structured intermediate code models."
cover: plug.png
---

![plug](plug.png)

*Building on [Part 1](../babylon_1_mlir_tosa/), this article focuses on OpenJDK Babylon’s **Code Reflection API**, the foundational mechanism that allows Java methods to be represented, analyzed, and transformed as structured intermediate code models within the JVM. Code Reflection is the infrastructure that enables Babylon’s compiler-style capabilities entirely in Java, without external toolchains.*

<!--more-->

---

## Scope and Intent

This post focuses exclusively on Babylon’s Code Reflection API. It explains how code models capture method bodies, how they can be inspected and transformed, and what this enables for Java tooling. We do not discuss MLIR, TOSA, or external compiler infrastructures here. Code reflection extends Java’s reflective capabilities beyond types and metadata, exposing method bodies as structured, analyzable representations suitable for transformation, analysis, and experimentation.

---

## From Reflection to Code Models

In standard Java, reflection lets you inspect structural aspects of a class: the methods it declares and their signatures. What it does not expose is what those methods do internally — the operations they perform, how values flow through them, and how control flows between different parts of the method body.

Project Babylon addresses this limitation by introducing code models. A code model is a representation of a method body that sits between Java source and bytecode. Unlike the compiler’s AST, which contains a great deal of syntactic detail, and unlike JVM bytecode, which has already flattened many of the high‑level constructs, a code model captures the essential structural and semantic information of a method in a form suitable for analysis and transformation. Babylon’s code model preserves type information and control structure, and it organizes the method body into a tree of operations, bodies, blocks, and values. This design is influenced by intermediate representations used in modern compilers, but it is tailored to be accessible and manipulable directly from Java code at runtime. Code models retain enough semantic richness to support meaningful analysis and transformation, while avoiding the syntactic verbosity of compiler ASTs and the semantic impoverishment of raw bytecode.

---

## Opting In with `@Reflect`

Code reflection is explicit. Only methods annotated with `@Reflect` are eligible:

```java
@Reflect
private double myFunction(int value) {
    return Math.pow(value, 2);
}
```

When the compiler encounters this annotation, it records sufficient metadata in the class file to reconstruct the method body as a code model at runtime. Methods not annotated remain unaffected, preserving Java’s existing compilation and execution characteristics.

---

## Obtaining a Code Model

At runtime, standard Java reflection is used only to *identify* the method:

```java
CoreOp.FuncOp codeModel = Op.ofMethod(method).get();
```

The transition from `Method` to `FuncOp` marks the handoff from Java reflection to Babylon code reflection. From this point onward, the program operates on a Babylon-defined IR rather than Java source or JVM bytecode.

`CoreOp.FuncOp` represents the entire method, including its parameters, body, and return semantics.

---

## Inspecting the IR

For debugging and exploration, Babylon provides a textual form:

```java
System.out.println(codeModel.toText());
```

This textual representation exposes the structure of the method in terms of operations and control flow. While primarily intended for diagnostics, it also serves as a useful mental model when learning how Babylon represents Java semantics internally.


```
func @loc="70:5:file:openjdk-babylon-mlir-tosa/cr-examples/samples/src/main/java/oracle/code/samples/HelloCodeReflection.java" @"myFunction" (%0 : java.type:"oracle.code.samples.HelloCodeReflection", %1 : java.type:"int")java.type:"double" -> {
    %2 : Var<java.type:"int"> = var %1 @loc="70:5" @"value";
    %3 : java.type:"int" = var.load %2 @loc="72:25";
    %4 : java.type:"double" = conv %3 @loc="72:16";
    %5 : java.type:"int" = constant @loc="72:32" @2;
    %6 : java.type:"double" = conv %5 @loc="72:16";
    %7 : java.type:"double" = invoke %4 %6 @loc="72:16" @java.ref:"java.lang.Math::pow(double, double):double";
    return %7 @loc="72:9";
};
```

---

## SSA Transformation

While Babylon’s code model is a structured, analyzable representation of a Java method, it is not initially in SSA form. In other words, variables in the code model can be assigned multiple times, and the flow of values can be implicit or distributed across different operations and blocks. This reflects the fact that the code model is designed to closely mirror the original Java semantics, including Java’s typical patterns of mutable variables, loops, and reassignments. Keeping the initial code model closer to the source makes it easier to perform analyses that need to respect original Java semantics, such as reasoning about side effects or field accesses, and simplifies the mapping back to the original code if needed.

Static Single Assignment (SSA) is a compiler-oriented transformation that changes this by ensuring each value is assigned exactly once and every variable is uniquely versioned. SSA makes data dependencies explicit, simplifies optimization and analysis, and is the standard form used in modern compilers because it allows easier reasoning about value flow, liveness, and transformations. In Babylon, the SSA pass transforms the source-like code model into this more analysis-friendly form:

```java
CoreOp.FuncOp ssaCodeModel = SSA.transform(codeModel);
```

After this pass, every variable assignment is unique, control flow is made explicit via block parameters, and the representation is now suitable for advanced analyses, transformations, or lowering to bytecode or other backends. The initial code model remains faithful to the original Java semantics, while SSA provides a canonical form optimized for compiler-style reasoning.

```
func @loc="70:5:file:openjdk-babylon-mlir-tosa/cr-examples/samples/src/main/java/oracle/code/samples/HelloCodeReflection.java" @"myFunction" (%0 : java.type:"oracle.code.samples.HelloCodeReflection", %1 : java.type:"int")java.type:"double" -> {
    %2 : java.type:"double" = conv %1 @loc="72:16";
    %3 : java.type:"int" = constant @loc="72:32" @2;
    %4 : java.type:"double" = conv %3 @loc="72:16";
    %5 : java.type:"double" = invoke %2 %4 @loc="72:16" @java.ref:"java.lang.Math::pow(double, double):double";
    return %5 @loc="72:9";
};
```

---

## Executable Semantics

A key property of Babylon’s IR is that it preserves executable semantics. The code model is not merely descriptive; it represents valid Java behavior that can, if needed, be interpreted or lowered to bytecode. This makes it suitable for validation, experimentation, and programmatic analysis, allowing developers to reason about method behavior in a structured form without departing from the Java platform.

---

## Traversing the IR Tree

Babylon provides access to the full hierarchy of elements within a code model, which allows tools and developers to inspect the internal structure of a method in detail. Each code element maintains a reference to its parent, making it possible to reconstruct the entire IR tree from any point in the model. This hierarchical access is valuable for a range of use cases, including visualization, custom analyses, or the development of transformation passes that need to reason about relationships between operations, blocks, and higher-level constructs. The API allows iteration over all elements of a code model, giving developers a flexible mechanism to explore the method body programmatically and understand both data and control flow patterns without resorting to bytecode or source parsing.

```java
codeModel.elements().forEach(e -> { ... });
```

---

## Implications of Code Reflection

The introduction of code reflection in Babylon significantly expands the range of tooling and programmatic capabilities available in Java. By exposing method bodies as structured, analyzable representations, code reflection enables semantic program analysis that was previously difficult or impossible using standard reflection. It facilitates source-to-source or IR-to-IR transformations entirely within Java, supporting domain-specific abstractions that can be implemented as libraries rather than external compilers. This infrastructure also opens the door to runtime experimentation and specialization, allowing developers to perform dynamic analyses or adapt execution strategies based on program behavior. Crucially, all of these capabilities are delivered as part of the Java platform itself, without relying on compiler plugins or non-standard toolchains, ensuring that they integrate seamlessly with existing development and deployment practices.

---

## Summary

Project Babylon’s Code Reflection API exposes Java method bodies as structured, analyzable IR. The example presented here demonstrates the full lifecycle: reflection, inspection, SSA transformation, and the preservation of executable semantics.

This capability moves Java closer to a model where *programs can reason about and transform their own semantics*, using APIs that follow established compiler principles while remaining idiomatic to the platform.
