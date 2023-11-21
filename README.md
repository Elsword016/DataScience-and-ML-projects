# DataScience_portfolio
Some projects that I have done in free time. I am largely self-taught so this repo, is a showcase of my learning.

## 1. CLIP-Seg + SAM segmentation
Implementation of simple panoptic segmentation using CLIPSegmentation model (https://huggingface.co/blog/clipseg-zero-shot) and then refining the low-resolution mask with Segment Anything(https://segment-anything.com/) by prompting it with points sampled within the low-res mask from CLIPSeg.

![CLIPsam](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/cda9759b-1c70-40af-b4f3-25a818d6c89d)

## Segment Anything model prompting ##
Trying out Segment Anything model from MetaAI with bounding box prompts. In this case, we give the bounding box coordinates to the model of the ROI and generate the a high-quality segmentation.
- Automatic mask generation 
![autogen](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/7fdb48e5-b7d5-4a84-9b0b-8e01031cb7f7)

- Bounding box prompt for mask generation 
![bbox_prompt](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/059a8c58-6c21-4467-acb4-d22d667ae712)
