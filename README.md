# Introduction
This project implements the task of Lesion Boundary Segmentation in ISIC 2018. We name the algorithm as *IVU: Improved VGG-Unet model in ISIC 2018*.

In the recognition of melanoma, deep learning plays an important role today. In this algorithm, we propose a lesion segmentation model based on VGG and UNET. The model receives dermatological pathological images and produces a binary lesion segmentation image. We participate in ISIC 2018 challenge and get a validation jaac score of 0.775. We even get a higher score of 0.8213 with model trained on 2017 dataset. In this paper, we first analyze the advantage of Unet and VGG structure, and then carry out some experiments based on them. For a relative higher score, we observe images whose scores are extremely high and low, and try to improve the model with multi-loss task and data augmentation. At last, we make a solid suggestion
of transferring images into HSL color space.

# Related Link
Detailed description of the algorithm can be found in link: [ISIC 2018 Phases](https://challenge.kitware.com/#challenge).