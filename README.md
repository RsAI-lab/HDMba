<div align="center">
<h1>HDMba</h1>
<h3>HDMba: Hyperspectral Remote Sensing Imagery Dehazing with State Space Model</h3>

[Hang Fu](https://hang-fu.github.io/), [Genyun Sun](https://ocean.upc.edu.cn/2019/1107/c15434a224792/page.htm), Yinhe Li, [Jinchang Ren](https://scholar.google.com.hk/citations?user=Vsx9P-gAAAAJ&hl=zh-CN), [Aizhu Zhang](https://ocean.upc.edu.cn/2019/1108/c15434a224913/page.htm), Cheng Jing, [Pedram Ghamisi](https://www.ai4rs.com/)


ArXiv Preprint ([arXiv 2401.09417](https://arxiv.org/abs/2401.09417)))


</div>


#



## Abstract
Haze contamination in hyperspectral remote sensing images (HSI) can lead to spatial visibility degradation and spectral distortion. Haze in HSI exhibits spatial irregularity and inhomogeneous spectral distribution, with few dehazing networks available. Current CNN and Transformer-based dehazing meth- ods fail to balance global scene recovery, local detail retention, and computational efficiency. Inspired by the ability of Mamba to model long-range dependencies with linear complexity, we explore its potential for HSI dehazing and propose the first HSI Dehazing Mamba (HDMba) network. Specifically, we design a novel window selective scan module (WSSM) that captures local dependencies within windows and global correlations between windows by partitioning them. This approach improves the ability of conventional Mamba in local feature extraction. By modeling the local and global spectral-spatial information flow, we achieve a comprehensive analysis of hazy regions. The DehazeMamba layer (DML), constructed by WSSM, and residual DehazeMamba (RDM) blocks, composed of DMLs, are the core components of the HDMba framework. These components effec- tively characterize the complex distribution of haze in HSIs, aid- ing in scene reconstruction and dehazing. Experimental results on the Gaofen-5 HSI dataset demonstrate that HDMba outperforms other state-of-the-art methods in dehazing performance.


<div align="center">
<img src="assets/vim_teaser_v1.7.png" />
</div>



## Train Your Vim

`bash vim/scripts/pt-vim-t.sh`

## Train Your Vim at Finer Granularity
`bash vim/scripts/ft-vim-t.sh`

## Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |

**Notes:**
- <sup>+</sup> means that we finetune at finer granularity with short schedule.
## Evaluation on Provided Weights
To evaluate `Vim-Ti` on ImageNet-1K, run:
```bash
python main.py --eval --resume /path/to/ckpt --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /path/to/imagenet
```
## Acknowledgement :heart:
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.

## Citation
If you find Vim is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @article{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Lianghui Zhu and Bencheng Liao and Qian Zhang and Xinlong Wang and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}
```
