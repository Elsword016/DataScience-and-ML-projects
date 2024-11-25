#  Simulating brainbow-like images 
 
This repository attempts to replicate the simulation based on the algorithm in the paper ***Automated scalable segmentation of neurons from multispectral images***, Sümbül et al. (2016)

## Motivation 
Brainbow is a process by which individual neurons in the brain can be distinguished from neighbouring neurons using fluorescent proteins. By randomly expressing different ratios of red, green, and blue derivatives of green fluorescent protein in individual neurons, it is possible to flag each neuron with a distinctive color. This process has been a major contribution to the field of neural connectomics.

So having a way to simulate/generate rainbow-like multispectral images is quite helpful, especially for trying out multi-colour image segmentation. The paper doesn't accompany an open-source implementation thus I tried to write one on my own.

### Algorithm overview:

![Screenshot 2024-11-20 135236](https://github.com/user-attachments/assets/1c6c9ce2-4486-4c7f-b5e9-83f69e523dfd)

From the paper we get this pseudo-code describing the algorithm shown in the paper. The code generates Brainbow image stacks from volumetric reconstructions of single neurons. For simplicity,  they model the neuron colour shifts by a Brownian noise component on
the neuron morphology, and the background intensity by a white Gaussian noise component.

## Working 

So the code inputs the morphology of the neurons in the standard `.swc` file. Then, a binary image volume based on a given size from the morphology is created. The generated volumetric image is used as input for the simulator.

![image](https://github.com/user-attachments/assets/f3c2761d-7e0c-4f50-925c-a6221d0c4108)

![image](https://github.com/user-attachments/assets/293063bd-f081-4d23-8b87-3bc010aa9e3b)

The image above shows the input data which is morphology and then generates a 5 colour channel images. I randomly added the same morphology with augmentations to add more variability to the image.

### Simulator 

Following are the parameters to the simulator. The simulator code is in the notebook `brainbow_simulator.ipynb` 

```bash
stack_shape = (100, 200, 200)  # 3D stack dimensions (z, y, x)
num_neurons = 10               # Number of neurons to simulate
num_channels = 5              # Number of color channels
noise_std = 0.05              # Background noise standard deviation
color_shift_std = 0.1         # Color variability standard deviation
saturation_level = 1.0        # Maximum voxel intensity 
```

## Citation 

```bash
@article{sumbul-2016,
	author = {Sümbül, Uygar and Roossien, Douglas and Chen, Fei and Barry, Nicholas and Boyden, Edward S. and Cai, Dawen and Cunningham, John P. and Paninski, Liam},
	journal = {Neural Information Processing Systems},
	month = {12},
	pages = {1920--1928},
	title = {{Automated scalable segmentation of neurons from multispectral images}},
	volume = {29},
	year = {2016},
	url = {https://papers.nips.cc/paper/6549-automated-scalable-segmentation-of-neurons-from-multispectral-images.pdf},
}
```


  
