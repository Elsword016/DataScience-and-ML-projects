## Simple background removal using the Depth-Pro model

A simple way to remove background from images from the depth map obtained from the [Depth-pro](https://huggingface.co/apple/DepthPro) by Apple. 

![bg_remove](https://github.com/user-attachments/assets/93ef6051-8ac5-4785-bc11-a4832a426df7)

Since the model returns the depth map and focal length, I calculated `disparity`. Disparity refers to the difference in image location of an object seen by the left and right eyes, or two cameras in a stereo imaging setup. 
In this case, we get depth directly with a single image depth estimation model. However, converting to disparity can still be useful for visualization and certain processing tasks.

In the context of background removal, using disparity instead of depth can sometimes provide better results because:

- The disparity values for foreground objects (closer to the camera) will be higher and more distinct from the background.
- The range of disparity values is often more suitable for simple thresholding operations.

However, the effectiveness still depends on the quality of your initial depth estimation and the specific characteristics of your scene.

Please look at the Depth-pro GitHub repo for installation instructions: [Depth-pro code](https://github.com/apple/ml-depth-pro) 

### Citation 

```
@article{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun}
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  journal    = {arXiv},
  year       = {2024},
}
```

