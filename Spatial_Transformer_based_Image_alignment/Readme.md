## Iterative Affine Transform using a Spatial Transformer network

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image to enhance the model's geometric invariance. For example, it can crop a region of interest, scale and correct the orientation of an image.

![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/6687f5ea-3fba-4125-a8cf-1bd5fdd208a3)

Here, I tackled an image registration problem to compute the suitable affine transforms between a fixed and moving image. The model is initialized with an identity transformation matrix `[1, 0, 0, 0, 1, 0]` and the model's job is to refine this matrix to get the suitable final transformation matrix. I used **Normalized Cross Correlation (NCC)** as the loss function as this is one of the standards used in un supervised image registration and we try to minimize this function.

## Results



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
