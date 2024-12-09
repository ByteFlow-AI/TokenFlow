# 1. TokenFlow

### Preparation

Run all tokenizer training/evaluation commands under the tokenflow folder:

```
cd tokenflow
```

### Training

For training the three variants of TokenFlow, modify the `tokenflow/scripts/train_vq.sh` accordingly and run

```
bash scripts/train_vq.sh
```

For decoder finetuning, run

```
bash scripts/train_vq_finetune_decoder.sh
```

For enhanced decoder finetuning, run

```
bash scripts/train_vq_finetune_enhanced_decoder.sh
```

### Evaluation

For evaluation of the reconstruction quality (rFID, PSNR, SSIM) of the released tokenizer, use:
```
bash scripts/reconstruction_vq_siglip.sh

bash scripts/reconstruction_vq.sh
```


# 2. Text-to-image Generation with TokenFlow

### Preparation

 - Dowload the pretrained 'clipb' version tokenizer from [here](https://huggingface.co/ByteFlow-AI/TokenFlow) and put it in the `TokenFlow/pretrained_ckpts/` folder.

- Install the enviroment package accodrding the overall requirements.


### Text to image generation inference
```
bash scripts/run_t2i.sh
```


### Training
 Setup your data and dataset code, we give an example dataset in [dataset.py](t2i/llava_t2i/dataset/dataset.py) for reference. Then run:

```
bash scripts/train_t2i.sh
```

# 3. Multimodal Understanding with TokenFlow

Our multimodal understanding code is slightly modified from [LLaVA](https://github.com/haotian-liu/LLaVA). We greatly appreciate their excellent work.

### Preparation

#### Install Package
```
cd i2t
pip3 install -e . 
pip3 install -e ".[train]" 
```
#### Data Prepare

Data of LLaVA 1.5: Please refer to [LLaVA](https://github.com/haotian-liu/LLaVA) for llava data preparation.

Cambrian data: Download the [Cambrian-Alignment](https://huggingface.co/datasets/nyu-visionx/Cambrian-Alignment/) and [Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M), and specify the data paths in `i2t/scripts/v1_5/train.sh`.

### Training

The global batch sizes for the pretraining stage and the finetuning stage are 256 and 128, respectively.
```
bash i2t/scripts/v1_5/train.sh  # pretrain and finetune 
```


### Evaluation

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval.git) to evaluate the model.