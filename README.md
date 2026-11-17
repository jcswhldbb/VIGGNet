# VIGGNet: A Vision Graph Neural Network–Guided Network for Medical Ultrasound Image Segmentation

This repository accompanies the paper **"VIGGNet: A Vision Graph Neural Network–Guided Network for Medical Ultrasound Image Segmentation."**

## Datasets

The experiments are conducted on four publicly available ultrasound datasets, covering two segmentation tasks:

| Dataset | Task |
| --- | --- |
| DDTI | Thyroid nodule segmentation |
| TN3K | Thyroid nodule segmentation |
| BUSI | Breast lesion segmentation |
| BUS-BRA | Breast lesion segmentation |

## Repository structure

- [`models/VIGGNet/VIGGNet.py`](models/VIGGNet/VIGGNet.py): VIGGNet model implementation.
- [`models/VIGGNet/gcn_lib/`](models/VIGGNet/gcn_lib/): graph convolution operators used by the model, including k-NN graph construction, graph convolution and position embedding.

The data split files used in this paper will be released in this repository.
