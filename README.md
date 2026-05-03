# DiDA
Official implementation of paper "Towards Unsupervised Domain Bridging via
Image Degradation in Semantic Segmentation" (NeurIPS 2025)

*Wangkai Li, Rui Sun, Huayu Mai and Tianzhu Zhang*

![Poster](NIPS25_DiDA_poster.png)

## Training

To train DiDA, run:

```bash
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dida.py