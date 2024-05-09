import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import exposure
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage as ndi
from skimage.measure import regionprops
from tqdm.notebook import tqdm

class PupilSegmentation(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=2, n_clusters=5):
        self.sigma = sigma
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        return self
    
    def preprocess(self,X):
        x = rgb2gray(X)
        x = gaussian(x, sigma=self.sigma)
        x = exposure.equalize_adapthist(x, clip_limit=0.03)
        return x
    
    def segment(self,X):
        x = self.preprocess(X)
        pix = x.reshape(-1,1)
        km = KMeans(n_clusters=self.n_clusters).fit(pix)
        labels = km.labels_.reshape(x.shape)
        mask =labels==np.argmin(km.cluster_centers_) # the darkest cluster
        mask = morphology.binary_closing(mask, morphology.disk(5))
        mask = morphology.binary_opening(mask, morphology.disk(5))
        mask_labeled, num_labels = ndi.label(mask)
        props = regionprops(mask_labeled)
        areas = [p.area for p in props]
        largest_region = np.argmax(areas)
        mask_largest = mask_labeled == largest_region + 1
        mask_filled = ndi.binary_fill_holes(mask_largest)
        centroid = ndi.center_of_mass(mask_filled)
        return mask_filled, centroid
    
    def track_pupil(self, frames):
        masks = []
        centroids = []
        
        for i in tqdm(range(frames.shape[0])):
            mask, centroid = self.segment(frames[i])
            masks.append(mask)
            centroids.append(centroid)
        
        return np.array(masks), np.array(centroids)
    
    def cumulative_dist(self,centroids):
        distance = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        return np.cumsum(distance)
    
    def save_results(self, masks, centroids, distance, output_path):
        np.save(output_path + 'masks.npy', masks)
        np.save(output_path + 'centroids.npy', centroids)
        np.save(output_path + 'distance.npy', distance)

    def visualize_results(self, centroids, cumulative_distances):
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 8))
        
        ax1.plot(centroids[:, 1], centroids[:, 0], marker='o', linestyle='-')
        ax1.set_title('Centroid Trajectory')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        ax2.plot(centroids[:, 1], marker='o', label='X')
        ax2.plot(centroids[:, 0], marker='o', label='Y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Coordinate')
        ax2.set_title('Centroid Trajectory (X and Y)')
        ax2.legend()
        
        ax3.plot(cumulative_distances, marker='o', linestyle='-')
        ax3.set_title('Cumulative Distance')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Cumulative Distance (pixels)')
        
        plt.tight_layout()
        plt.show()
    
    def process(self, X, output_path):
        masks, centroids = self.track_pupil(X)
        distance = self.cumulative_dist(centroids)
        self.save_results(masks, centroids, distance, output_path)
        self.visualize_results(centroids, distance)
        return masks, centroids, distance

