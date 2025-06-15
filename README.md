<div align="center">


TeleTron
===========================

<h4>To Pioneer Long-Context Multi-Modal Transformer Training</h4>

[![version](https://img.shields.io/badge/release-0.1.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache2.0-blue)](./LICENSE)

<div align="left">

## ⏱️Speed Benchmark 

| High Training Throughput | Long Training Contexts  |
|:---:|:---:|
| <img src="assets/efficiency.png" width="400" alt="Training Efficiency"> | <img src="assets/efficiency_max_training_frames.png" width="400" alt="Max Training Frames"> |

^ Experiments conducted on HunyuanVideo model training using the first released version (2025/05/16).

## 🔥News

- **2025/06/13**: TeleTron accelerates HunyuanVideo training by 30%+ by Distributed Multi-Modal Encoders!
- **2025/05/16**: TeleTron first release with code for HunyuanVideo full-parameter training and inference! [[Zhihu]](https://zhuanlan.zhihu.com/p/1907030055512671098) [[WeChat]](https://mp.weixin.qq.com/s/Ie1NulNlUmzqSCRCFAXy7Q)

## 📖Introduction

TeleTron features flexible parallel strategy and fused cuda kernels to best facilitate **long-context**, **efficient** and **flexible** training of multi-modal transformer models.

* 📜 **Long-Context Training**</br>
  TeleTron leverages mixed parallel strategy, activation checkpointing and fused cuda kernels at the same time to optimize GPU memory usage, so as to train [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) with more than 30 seconds of video clips in 720P.
* 🚀 **Efficient at Scale**</br>
  Through CUDA optimization and distributed training techniques, TeleTron can achieve higher throughput than general Transformer training frameworks, especially at a large scale.
* 🛠️ **Flexible in Application**</br>
  Training with a variety of video sequence length and model size, TeleTron supports flexible adjustment of parallel strategy among data parallel, context parallel, and/or tensor parallel.

TeleTron has released code for HunyuanVideo I2V fine-tuning and has been supporting **[TeleAI VAST](https://arxiv.org/abs/2412.16677v1)** (Video As Storyboard from Text) on high-resolution video generation training (code to be released). 

## ⚡️QuickStart

### Installation

To save efforts on environment setup, it is recommended using [nvcr](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)'s 24.10-py3 container image. 

```
# pull docker image
docker pull nvcr.io/nvidia/pytorch:24.10-py3

# start docker container
sudo docker run --gpus all -itd --shm-size 512G --name teletron  nvcr.io/nvidia/pytorch:24.10-py3 /bin/bash

# enter the container
sudo docker exec -it teletron /bin/bash
```

In the docker container, follow the script below to setup TeleTron.

```
# get TeleTron
git clone git@github.com:Tele-AI/TeleTron.git --recurse-submodule

# install requirements
pip install -r requirements.txt

# install TeleTron fused kernels 
cd teletron_op && bash install.sh && cd -
```

### Sanity Check

The script below will run a tiny version of HunyuanVideo with fake data. It serves as a sanity check for that the environment is correctly set up.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=12345 bash examples/hunyuanvideo/run_unified_sanity_check.sh 1 1
```

### Training

* single node training

```
bash examples/hunyuanvideo/run_unified.sh 2 2 9
```

Note that the numbers "2 2 9" above denotes TP size, CP size, and number of frames respectively.  The default video resolution is 720P, and you may also alter training video resolution by adding `--video-resolution {width} {height}`  to the training arguments in the shell script `run_unified.sh`. 

* Multi-node training

Run the script below respectively on 4 * 8 H800 nodes and 129-frame 720P training will be initiated. Note that for full finetuning you still need to [download](https://huggingface.co/tencent/HunyuanVideo/tree/main) and [convert](./teletron/convert_ckpt/hunyuan/convert_ckpt.md) HunyuanVideo pretrained weights.

```
bash examples/hunyuanvideo/run_unified.sh 1 4 129
```


## ✨Features

- [x] Ulysses Context Parallel
- [x] Tensor Parallel 
- [x] AdaLayerNorm fused kernel
- [x] RmsNorm fused kernel
- [x] Distributed multimodal encoders
- [ ] [Unified Sequence Parallel](https://arxiv.org/abs/2405.07719) 

## Acknowledgement

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* [Diffusers](https://github.com/huggingface/diffusers)
* [yunchang](https://github.com/feifeibear/long-context-attention)
* [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
* [Koala-36M](https://github.com/KwaiVGI/Koala-36M)
* [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)

## License

[Apache 2.0 License](./LICENSE)

