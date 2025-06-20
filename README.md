# SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning

Welcome to the official code repository for [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning **(ICLR 2025, Oral)**](https://openreview.net/pdf?id=5U1rlpX68A).

If you find this code useful in your research then please cite  
```bibtex
@inproceedings{
wu2025sdlora,
title={{SD}-Lo{RA}: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning},
author={Yichen Wu and Hongming Piao and Long-Kai Huang and Renzhen Wang and Wanhua Li and Hanspeter Pfister and Deyu Meng and Kede Ma and Ying Wei},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5U1rlpX68A}
}
``` 

## 👀 Introduction
![SD-LoRA](imgs/intro.jpg)

- SD-LoRA introduces a decoupled learning strategy for the magnitude and direction of LoRA components to achieve scalable continual learning without rehearsal of huge sample features.
- It demonstrates a strong stability-plasticity trade-off by converging to overlapping low-loss regions across sequential tasks, supported by empirical and theoretical analysis.
- SD-LoRA and its two variants enable end-to-end optimization and efficient inference without component selection, achieving state-of-the-art performance on multiple CL benchmarks and foundation models.

## 📜 Results
![SD-LoRA](imgs/results1.jpg)
![SD-LoRA](imgs/results2.jpg)
- To run the experiments, download the datasets to /data/ and execute:
   ```bash
  bash run.sh
- For your convenience, we have provided the running logs in the log file, where you can find detailed performance results for all streaming tasks.



## 🙏 Acknowledgement
This repo is built upon the following projects:

* [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)
* [PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
