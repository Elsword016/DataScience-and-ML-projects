# Projects
Some projects to showcase my learning, particularly focussing on fine-tuning models on biological data and image processing in general. Some might be related to experiment data for which I won't share the data for obvious reasons. Most of the projects focus on Computer Vision since I work with image data a lot, but I will try to put more projects related other domains such as NLP.

## 1. [Segment Anything model prompting](SAM_prompting) ##
Trying out Segment Anything model from MetaAI with bounding box prompts. In this case, we give the bounding box coordinates to the model of the ROI and generate the a high-quality segmentation.
- Automatic mask generation 
![autogen](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/7fdb48e5-b7d5-4a84-9b0b-8e01031cb7f7)

- Bounding box prompt for mask generation 
![bbox_prompt](https://github.com/Elsword016/Road-to-learning-ML/assets/29883365/059a8c58-6c21-4467-acb4-d22d667ae712)

## 2. [Blood Cell Classification with ViT from scratch](ViT_from_scratch-BloodCellClassifier)
Using Pytorch I implemented a Vision Transformer totally from scratch, similar to the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). I closely followed [Uva deep learning course](https://uvadlc.github.io/) explanation and how to define pytorch lightning training setup.
| ![ViT-arch](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/21bb6c70-a0a0-46b3-b1ab-c6a39e6100eb) | Prediction ![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/4e24ebd9-88c6-4e12-9f2c-7ba53c761634) 
|-------------------------|-------------------------|

## 3. [Segmenting Retina blood vessels using a finetuned Segformer model](RetinaVessels_segmentation_SegFormer)
Segmentation using Segformer base model from HuggingFace on Retina blood vessel dataset with a simple gradio interface for inferencing

Dataset obtained from Kaggle - [Blood Vessel Data](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)


![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/92f1efaf-8aaf-444d-a9bd-c2f636db167d)

## 4. [Dendritic spine segmentation using Segformer with LoRA Adapters](Spine_Segmentation)
Segmentation of dendritic spines from 2-photon microscopy images. I used the base Segformer model and then converted it into a LoRA model using the PEFT library to reduce the number of parameters.

More details here-> [ReadMe](Spine_Segmentation/Readme.md)

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/be50ee43-ffe9-45a6-83e6-882935776142)

## 5. [Spine segmentation with SAM using point prompts](Spine_Segmentation)
Spine segmentation by finetuning [Segment Anything](https://ai.meta.com/research/publications/segment-anything/) model using points as prompts rather than bounding boxes. 

For each of the ground truth masks, points (n=100) are sampled from the non-zero pixel positions
then forward it to the prompt encoder. 

I used the HuggingFace 🤗 [SamModel](https://huggingface.co/facebook/sam-vit-base) class.

(It will be a nice experiment to see the quality of detection using bounding box prompt vs points)

**Dataset** - My own spine images
![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/5222d82c-7373-4cbc-bf4b-ac6f02a556b4)

## 7. [A quick eye-tracking solution](Eye_tracking)
A simple pupil segmentation followed by centroid tracking. Segmentation is done using K-means and simple image-processing tricks. It's not SOTA, but I came with it in 4 hours for a competition.

**Dataset** - proprietary 


| ![segmentation](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/9aab427b-de83-4995-bebf-237ff2090b55) | ![output_track](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/d204341a-ca4c-4d13-a802-f544180607a6)
|-------------------------|-------------------------|

## 8. [Zero-shot Panoptic Eye Segmentation](Zero_shot_eye_tracking)
A simple panoptic segmentation of the pupil for eye-tracking combining CLIP-Seg and SAM models. An improved version of the previous simple pupil segmentation I did for a competition.

Dataset: Private
![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/9d7bcf8a-7631-4e16-bd97-f18285968de8)

## 9. [BELKA binding prediction](BELKA_prediction)

My approach for the Kaggle competition [Leash Bio](https://www.kaggle.com/competitions/leash-BELKA/overview) to predict the binding nature between a protein and a molecule. Using simple Xgboost with molecular fingerprints as input features and also fine-tuning BERT model from HuggingFace, by **tokenizing the protein and the smiles together** using a `[SEP]` token similar to what is done for question-answering datasets. Also comes with a small gradio interface for easy testing new inputs.

![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/590f2c09-0ba7-43a8-94ca-2485e7d29c8a)

## 10. [Iterative Affine Transform with Spatial Transformers](Spatial_Transformer_based_Image_alignment)

I tackled an image registration problem to compute the suitable affine transforms between fixed and moving images of blood vessels using **Spatial Transformers**. The model is initialized with an identity transformation matrix [1, 0, 0, 0, 1, 0] and the model's job is to refine this matrix to get the suitable final transformation matrix. I used Normalized Cross Correlation (NCC) as the loss function as this is one of the standards used in unsupervised image registration and we try to minimize this function.

<img width="525" alt="Screen_Shot_2020-07-19_at_5 48 34_PM_vFLk7jR" src="https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/91040969-6bb8-4ee3-835b-ef32c5401b13">

## 11. [Depth-Based Background Removal](Depth_based_background_removal)

A simple way to remove background from images using Apple's Depth-pro model.

![bg_remove](https://github.com/user-attachments/assets/6a436c07-c6e3-4422-8144-a4174bee6808)







