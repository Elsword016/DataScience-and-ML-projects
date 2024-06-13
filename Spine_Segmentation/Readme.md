## Spine Segmentation from 2-photon microscopy images using finetuned SegFormer with LoRA config
### Dataset info:
M. U. Ghani, F. Mesadi, S. D. Kanik, A. O. Argunsah, A. Hobbiss, I. Israely, D. Unay, T. Tasdizen, and M. Cetin, **"Shape and appearance features based dendritic spine classification," Journal of Neuroscience Methods**. 
[GitHub page](https://github.com/mughanibu/Dendritic-Spine-Analysis-Dataset)

A fully annotated dataset of Two-Photon Laser Scanning Microscopy (2PLSM) images of three types of dendritic spines. We make a standard dendritic analysis dataset publicly available including raw data, manual annotations (segmentations), and manual labels. Manual labels and annotation (segmentations) are provided by a neuroscience expert.

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/399b0d0f-5295-458d-ad76-49655a64aa61)

### Model details:
I used the **Segformer-b0 model** from Nvidia from the HuggingFace library. Model library -> https://huggingface.co/nvidia/mit-b0

Since, I am GPU poor I have only RTX 3080 12G. So I used PEFT library to reduce the number of trainable parameters.

Params:
**Segformer-b0**

trainable params: 3714658 || all params: 3714658 || trainable%: 100.00

**Segformer-b0 with LoRA adapters**

trainable params: 526338 || all params: 4240996 || trainable%: 12.41

Results:

Sample input

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/473082a5-eb6c-49b9-aca9-f05440359d7d)

Predicted mask

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/1647ee3a-c73b-4b3a-b2c1-c769267da0d3)

To do: Tune the model with my own data and evaluate metrics and tune hyperparams
