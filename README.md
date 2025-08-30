# Implementations of LLM Watermarking Algorithms

This repository contains Python implementations of several recent watermarking algorithms for Large Language Models (LLMs). The goal is to provide clear, concise, and faithful replications of these schemes for research purposes.

***

## Algorithms Implemented

This library includes implementations of the following watermarking schemes:

* **Christ**: A detection and payload-embedding watermark based on arithmetic coding.

> Christ, M., Gunn, S., & Zamir, O. (2023). *Undetectable Watermarks for Language Models*. [arXiv:2306.09194](https://arxiv.org/abs/2306.09194).


* **OZ**: A robust, payload-embedding watermark that uses a dynamic error-correcting code to ensure message delivery.

> Zamir, O., et al. (2024). *Excuse me, sir? Your language model is leaking (information)*. [arXiv:2401.10360](https://arxiv.org/abs/2401.10360).


* **DISC**: A payload-embedding scheme based on a circular-shifted version of arithmetic coding that uses Gray codes for robustness.

> Kordi, Y., et al. (2025). *Multi-Bit Distortion-Free Watermarking for Large Language Models*. [arXiv:2402.16578](https://arxiv.org/abs/2402.16578).

***



