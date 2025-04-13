# SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning

Welcome to the official code repository for "[SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning **(ICLR 2025, Oral)**]([https://arxiv.org/abs/2406.01721](https://arxiv.org/pdf/2501.13198))".

## 👀 Introduction

- SD-LoRA introduces a decoupled learning strategy for the magnitude and direction of LoRA components to achieve scalable continual learning without rehearsal.
- It demonstrates a strong stability-plasticity trade-off by converging to overlapping low-loss regions across sequential tasks, supported by empirical and theoretical analysis.
- SD-LoRA and its two variants enable end-to-end optimization and efficient inference without component selection, achieving state-of-the-art performance on multiple CL benchmarks and foundation models.

## 📜 Results
- To run the experiments, simply execute:
   ```bash
  bash run.sh
- For your convenience, we have provided the running logs in the log file, where you can find detailed performance results for all streaming tasks.



## 🙏 Acknowledgement
This repo is built upon the following projects:

* [LoRA-ViT]([https://github.com/JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT))
* [PILOT]([https://github.com/sun-hailong/LAMDA-PILOT](https://github.com/JamesQFreeman/LoRA-ViT))
