---
title: "Are we quantum yet? The State of Practical Quantum Computing"
date: 2026-02-05
tags: [ "Quantum Computing", "MLIR" ]
description: "Quantum computing is no longer primarily limited by physics breakthroughs.
The real bottlenecks are increasingly in software, tooling, and systems integration."
cover: circle.jpg
---

![circle](circle.jpg)
*Photo by [Dynamic Wang](https://unsplash.com/de/@dynamicwang) on [Unsplash](https://unsplash.com/de/fotos/hintergrundmuster-vNCBkSX3Nbo)*

Quantum computing has been “five to ten years away” for about two decades now. Yet every few months, a new paper, breakthrough, or record-breaking qubit announcement rekindles the same question:

👉 **Are we quantum yet?**

If you’re familiar with the Rust ecosystem, you might recognize [Are we web yet?](https://www.arewewebyet.org/) and [Are we GUI yet?](https://areweguiyet.com/) trackers — a community-driven effort to measure whether Rust was ready to replace existing tooling in real-world production scenarios. The question wasn’t just about language features, but about ecosystem maturity, tooling, and practical usability.

Quantum computing sits in a surprisingly similar place today. The theory is powerful, the potential is enormous, but practical adoption still feels just out of reach.

Recently, I read the paper: [Computer Science Challenges in Quantum Computing: Early Fault-Tolerance and Beyond](https://arxiv.org/pdf/2601.20247)

The paper explores how the bottlenecks in quantum computing are shifting away from pure hardware limitations and toward computer-science challenges such as software stacks, algorithms, error correction, and system architecture.

In this post, I want to break down what this work contributes, why it matters, and — most importantly — whether it moves us any closer.

<!--more-->

---

The central thesis of the paper is simple but important:

> Quantum progress is no longer just a physics problem — it is increasingly a systems and computer science problem.

The authors argue that the field is transitioning from the NISQ era (Noisy Intermediate-Scale Quantum) toward early fault-tolerant quantum computing. In this new phase, systems will have:

- **Small numbers of logical qubits**  
- **Tight constraints on error rates**  
- **Strong dependence on classical control systems**  
- **Heavy integration across the full stack**

Instead of waiting for massive, fully fault-tolerant machines, the paper suggests that useful quantum computation will likely emerge gradually through early fault-tolerant systems, and that the ability to use these systems effectively depends on solving several major computer science challenges.

| Area | Core Question | Early Fault-Tolerant Challenges | Progress Signals |
|---|---|---|---|
| Algorithms & Cryptography | Which problems can gain from quantum speedups? | Advantages that hold under realistic assumptions<br>and vs classical methods | Problem classes with measurable quantum benefit<br>and clear baselines |
| Error Correction & Fault Tolerance | Can large-scale QEC be practical? | High overhead, decoding delays,<br>classical control costs | Automated code/decoder generation<br>and integrated low-latency pipelines |
| Software Stack | Can software run efficiently across hardware? | IRs and compilers supporting fault tolerance,<br>heterogeneity, and runtime dynamics | Hardware-agnostic programs,<br>transparent cost models,<br>verified transformations |
| Architecture & Systems | Can specialized machines give early usefulness? | Mapping limited logical qubits to real workloads | Full-stack co-designed experiments<br>showing application-level gains |

### Algorithms, Complexity, and Cryptography

> What problems actually benefit from quantum computing?

This sounds obvious, but it is arguably the hardest open question. While famous algorithms like Shor’s factoring algorithm demonstrate theoretical quantum advantage, the paper highlights that many quantum algorithms exist only in idealized theoretical models. Classical algorithms often “catch up” and eliminate expected quantum speedups. Demonstrating practical quantum advantage requires identifying realistic workloads and proving they remain hard for classical computers.

### Error Correction and Fault Tolerance

> Can quantum error correction scale?

Quantum hardware is extremely fragile. Logical qubits require many physical qubits for error protection. The paper highlights several major challenges:

- Massive space-time overhead from error correction  
- Decoding latency and classical processing costs  
- Lack of automated methods for designing codes and decoders  
- Need for integrated hardware/software co-design

One particularly interesting point is that the field must move beyond handcrafted error-correction schemes toward **automated, end-to-end error-correction workflows**. This is a huge engineering challenge — arguably similar to the evolution from hand-written assembly to optimizing compilers.

### Software Stack and Programming Models

> Can quantum software run efficiently across diverse hardware?

The paper argues that quantum computing currently lacks something classical computing takes for granted: portable and reliable software ecosystems. Major challenges include designing intermediate representations (IRs) for fault-tolerant computation, supporting heterogeneous hardware platforms, building compilers that incorporate error correction, and creating transparent cost models for quantum programs.

This part resonated strongly with me as someone working in [MLIR](/tags/MLIR) and compilers. The authors explicitly call out the need for verified transformations, portable programs, and runtime systems capable of handling hybrid classical-quantum execution.

In other words, quantum computing may need its own equivalent of **LLVM-scale infrastructure**.

### Architecture and Domain-Specific Systems

> Can specialized machines deliver early quantum usefulness?

Instead of waiting for universal quantum computers, the paper suggests that domain-specific quantum architectures may enable earlier real-world applications. This idea mirrors trends in classical computing, where specialized hardware like GPUs, TPUs, and accelerators often drive early breakthroughs. The key challenge is mapping scarce logical qubits to meaningful workloads while optimizing entire stacks from algorithms to hardware.

### Key Caution from the Paper ⚠️

One of the most valuable contributions of the paper is its caution against misinterpreting progress. It warns about:

- Over-interpreting small demonstrations  
- Prematurely standardizing toolchains or architectures  
- Oversimplified benchmarking metrics  
- Underestimating integration complexity

The authors emphasize that early fault-tolerant systems will likely produce **learning milestones** rather than definitive proof of quantum advantage.

### From Physics to Systems Engineering 🧭

Historically, quantum computing progress was dominated by improving qubit hardware. We are now entering a phase where:

- Hardware diversity is increasing  
- Error correction is becoming unavoidable  
- Classical control and software integration dominate performance  

**The future of quantum computing may look less like building better qubits and more like building better computing systems.**

### Are We Quantum Yet? ✅

👉 Not yet — but we are entering a new phase.  

The physics gave us possibility. Now **computer science has to make it usable**.
