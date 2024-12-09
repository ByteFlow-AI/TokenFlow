# TokenFlow🚀: Unified Image Tokenizer for Multimodal Understanding and Generation

<div align="center">

[![TokenFlow](https://img.shields.io/badge/Paper-TokenFlow-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2412.03069)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-TokenFlow-yellow)](https://huggingface.co/ByteFlow-AI)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green?logo=bytedance)](https://byteflow-ai.github.io/TokenFlow/)&nbsp;

</div>


## 🌿 Introduction

We present TokenFlow, a unified image tokenizer that bridges the long-standing gap between multimodal understanding and generation. 
TokenFlow introduce an innovative dual-codebook architecture that decouples semantic and pixel-level feature learning while maintaining their alignment through a shared mapping mechanism. 


<div align='center'>
<img src="./assets/radar.png" class="interpolation-image" alt="radar." height="50%" width="50%" />
</div>

TokenFlow excels in both multimodal understanding and image generation. For multimodal understanding, we surpass the flagship models such as LLaVA-1.5 and EMU3 by a large margin. For text-to-image generation, we also achieve comparable performance to SDXL in 256×256 resolution.

<div align='center'>
<img src="./assets/teasor.png" class="interpolation-image" alt="teasor." height="100%" width="100%" />
</div>

## 📰 News

**2024.12.9**:  Code and checkpoints are released.

**2024.12.5**:  🎉🎉🎉 TokenFlow is released! 🎉🎉🎉  See our [project page]() and [paper]() .


## ⚙️ Getting Started

See [GETTING_STARTED.md](./GETTING_STARTED.md) for detailed instructions of ***training*** and ***evaluation*** of (1) TokenFlow, (2) multimodal understanding model and (3) text-to-image generation model.


## 🤗 Checkpoints

**Text-to-Image Model**

<table>
  <tr>
    <th style="width: 150px;">Model Size</th>
    <th>Tokenizer Weight</th>
    <th>Model Weight</th>
  </tr>
  <tr>
    <td align="center">7B</td>
    <td align="center"><a href="https://huggingface.co/ByteFlow-AI/TokenFlow/blob/main/tokenflow_clipb_32k_enhanced.pt">TokenFlow</a></td>
    <td align="center"><a href="https://huggingface.co/ByteFlow-AI/TokenFlow-t2i">TokenFlow-t2i</a></td>
  </tr>
</table>

**Multimodal Understanding Model**

<table>
  <tr>
    <th style="width: 150px;">Language Backbone</th>
    <th>Tokenizer Weight</th>
    <th>Model Weight</th>
  </tr>
  <tr>
    <td align="center">Qwen-2.5-14B</td>
    <td align="center"><a href="https://huggingface.co/ByteFlow-AI/TokenFlow/blob/main/tokenflow_siglip_32k.pt">TokenFlow-XL</a></td>
    <td align="center"><a href="https://huggingface.co/ByteFlow-AI/TokenFlow-i2t">TokenFlow-i2t</a></td>
  </tr>
</table>


## 📑 Open-source Plan

- [X] Release the checkpoint of tokenizer, text-to-image model & multimodal understanding model.
- [X] Release the training & inference code for tokenizer.
- [X] Release the training & inference code for text-to-image generation.
- [X] Release the training & inference code for multimodal understanding.
- [ ] Release the single-scale version of TokenFlow.



## Acknowledgement

We thank the great work from [VAR](https://github.com/FoundationVision/VAR), [LlamaGen](https://github.com/FoundationVision/LlamaGen) and [LLaVA](https://github.com/haotian-liu/LLaVA).


## 📄 Citation

If our work assists your research, feel free to give us a star ⭐ or cite us using

```
@article{qu2024tokenflow,
  title={TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation},
  author={Qu, Liao and Zhang, Huichao and Liu, Yiheng and Wang, Xu and Jiang, Yi and Gao, Yiming and Ye, Hu and Du, Daniel K and Yuan, Zehuan and Wu, Xinglong},
  journal={arXiv preprint arXiv:2412.03069},
  year={2024}
}
```


## 🔥 Open positions
We are hiring interns and full-time researchers at the ByteFlow Group, ByteDance, with a focus on multimodal understanding and generation. If you are interested, please contact quliao1117@gmail.com.
