# A quick and easy way to segment pupil with simple tracking algorithm using cumulative distances of the centroids

I submitted it for a competition, a simple way to segment the pupil of the eye from videos using K-means and simple image processing methods. It doesn't aim to provide an optimal/SOTA solution but rather something I came up with in 4 hours. It will be nice to extend it to more real-time predictions. Maybe incorporating Zero-shot segmentation using CLIP-SAM and then refining it using SAM will improve it.

## Dependenices

```bash
matplotlib==3.8.3
numpy==1.26.4
scikit_learn==1.3.2
scipy==1.11.3
skimage
tqdm==4.66.1
```

## Files:

- `pupil_seg.py`: The main class in with easy APIs to segment and generate tracking results.
- `pupil_seg_api.ipynb`: Using the API for segmentation and visualizations
`
## Sample output:


 https://github.com/Elsword016/science_mltask/assets/29883365/49061bfe-ffcc-4ce7-b4ee-c05754df5252


