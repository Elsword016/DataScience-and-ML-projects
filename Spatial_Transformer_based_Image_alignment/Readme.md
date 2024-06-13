## Iterative Affine Transform using a Spatial Transformer network

This is a **Pytorch** implementation of [Spatial Transformer network](https://arxiv.org/abs/1506.02025) 

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image to enhance the model's geometric invariance. For example, it can crop a region of interest, scale and correct the orientation of an image.

![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/6687f5ea-3fba-4125-a8cf-1bd5fdd208a3)

## Background info

Here, I tackled an image registration problem to compute the suitable affine transforms between fixed and moving images of blood vessels. The model is initialized with an identity transformation matrix `[1, 0, 0, 0, 1, 0]` and the model's job is to refine this matrix to get the suitable final transformation matrix. I used **Normalized Cross Correlation (NCC)** as the loss function as this is one of the standards used in unsupervised image registration and we try to minimize this function.

## Results

The model takes in inputs of two images that are to be registered, and then using a suitable optimizer the ncc loss was minimized. Repeated iteration and experimenting with hyperparameters such as learning rate, epochs etc is required for proper optimization.
![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/ece35883-cf44-47e2-b825-5611b75ed163)

**The model also returns the final transformation matrix which then can be applied on other images.**

## Reference
```
@misc{jaderberg2016spatial,
      title={Spatial Transformer Networks}, 
      author={Max Jaderberg and Karen Simonyan and Andrew Zisserman and Koray Kavukcuoglu},
      year={2016},
      eprint={1506.02025},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
