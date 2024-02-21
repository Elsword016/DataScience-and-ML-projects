# DataScience_portfolio
Some projects that I have done in free time. I am largely self-taught so this repo, is a showcase of my learning.

## 1. CLIP-Seg + SAM segmentation
Implementation of simple panoptic segmentation using CLIPSegmentation model (https://huggingface.co/blog/clipseg-zero-shot) and then refining the low-resolution mask with Segment Anything(https://segment-anything.com/) by prompting it with points sampled within the low-res mask from CLIPSeg.

![CLIPsam](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/cda9759b-1c70-40af-b4f3-25a818d6c89d)

## 2. Segment Anything model prompting ##
Trying out Segment Anything model from MetaAI with bounding box prompts. In this case, we give the bounding box coordinates to the model of the ROI and generate the a high-quality segmentation.
- Automatic mask generation 
![autogen](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/7fdb48e5-b7d5-4a84-9b0b-8e01031cb7f7)

- Bounding box prompt for mask generation 
![bbox_prompt](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/059a8c58-6c21-4467-acb4-d22d667ae712)

## 3. EDA+Streamlit Dashboard - Tips prediction
Simple Exploratory data analysis and regression modelling to predict the amount of tips received by the waiter. I learned how to use streamlit to create a simple dashboard for data visualization and also use the models for prediction based on values obtained from the user. [Dashboard link](https://datascienceportfolio-kkcsplxxk3xdzrsabx5wi8.streamlit.app/)
![dash](https://github.com/Elsword016/DataScience_portfolio/blob/main/waiter_tips/Screenshot%202024-02-13%20at%2014.11.22.png)

## 4. Blood Cell Classification with ViT from scratch
Using Pytorch I implemented a Vision Transformer totally from scratch, similar to the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). I closely followed [Uva deep learning course](https://uvadlc.github.io/) explanation and how to define pytorch lightning training setup.
| ![ViT-arch](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/21bb6c70-a0a0-46b3-b1ab-c6a39e6100eb) | Prediction ![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/4e24ebd9-88c6-4e12-9f2c-7ba53c761634) 
|-------------------------|-------------------------|

## 5. Segmenting Retina blood vessels using a finetuned Segformer model
Segmentation using Segformer base model from HuggingFace on Retina blood vessel dataset

Dataset obtained from Kaggle - [Blood Vessel Data](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)

Gradio demo - [Retinopathy Confocal images](gradio_demo.ac)

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/92f1efaf-8aaf-444d-a9bd-c2f636db167d)

