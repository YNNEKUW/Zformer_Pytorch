# Zformer_Pytorch

## Table of Contents
- [Introduction](#Introduction)
- [Usage](#Usage)
- [FAQ](#FAQ)

## Introduction
According to ([2009.06732](https://arxiv.org/abs/2009.06732)), many Transformer-based models by the name of -formers have been proprosed to improve the efficiency during inference, especially by means of reducing the inference time. The main focus of most -formers is the modification of the attention modules since the dot-product operation in an attention module makes the square computation bottleneck of a Transformer model. In this repository, we implement the Zformer attention module. The module has been made compatible with [fairseq](https://github.com/pytorch/fairseq).
### Zformer architecture
We keep the original attention operation while modifying the projection matrices of Q,K and V. The projection matrices for K and V share the same weights and pool the sequence-length dimension of the input sequence by <em>$\beta$</em> (> 1). The projection matrix for Q pools the input sequence's hidden dimension by <em>$\alpha$</em> (> 1). Also, K is projected from Q instead of the input sequence of the attention module. Denote **n** as input sequence length and **d** as sequnce length. The computation complexity of attention probability is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/($\alpha\beta$))</em>, and that of the dot-product of attention probability with V is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/$\beta$)</em>.
The performance of the baseline Transformer model on the IWSLT 14 DE-EN translation task is 34.20 bleu score and our Zformer's performance is 32.09 bleu score while achieving substantial acceleration.
### Inference time comparison
The experiment is to measure the total inference time of the baseline and the proposed attention modules on an input tensor of shape (sequence length, batchsize, hidden dimension), while batchsize is set 40, we vary the sequence length as {128, 256, 512, 1024, 2048}, and hidden dimension as {512, 768, 1024}. The numbers in the table are inference time in seconds.
| Sequence length\$\alpha$$\beta$ | 512 | 768  |1024 |baseline  |
| ------------- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| 128        | 2.5553 | 1.9619 | 1.6746 | 3.9145 |
| 256        | 5.3080 | 3.9689 | 3.3009 | 8.8856 |
| 512        | 12.7100| 8.7503 | 6.8507 | 21.0929|
| 1024       | 34.3950| 21.6591| 15.6320| 58.1030|
| 2048       |105.3789| 60.5599| 40.2494|200.8014|
## Usage
For using the Zformer attention module, first do
```
git clone https://github.com/YNNEKUW/Zformer_Pytorch.git
pip install -r Zformer_Pytorch/requirement.txt
mv Zformer_Pytorch/zformer_pytorch .
```
and then
```python
from zformer_pytorch import Zformer
```
### Example
```python
import torch
from fairseq.modules.multihead_attention import MultiheadAttention
from zformer_pytorch import Zformer


hidden_dim = 512
n_heads = 4
batch_size = 40
length = 1024

baseline_attn = MultiheadAttention(hidden_dim, n_heads, self_attention=True).cuda()
test_input = torch.ones((length, batch_size, hidden_dim)).cuda()
dummy_out = baseline_attn(test_input, test_input, test_input)

xformer_attn = Zformer(hidden_dim, n_heads).cuda()
dummy_out = zformer_attn(test_input)
```

## FAQ
