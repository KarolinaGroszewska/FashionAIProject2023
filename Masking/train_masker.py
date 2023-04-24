

# pip install git+https://github.com/facebookresearch/segment-anything.git
# download default/vit_h checkpoint


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
sys.path.append("..")


# TrainMasker uses the pretrained SAM model to generate binary masks for a given image

class TrainMasker():
    def __init__(self, model='predictor'):
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        
        # device = "cuda"
        # self.sam.to(device=device)

        if model=='auto_generator':
            self.generator = SamAutomaticMaskGenerator(self.sam, output_mode='binary_mask')
        else:
            self.generator = SamPredictor(self.sam)
            
            
    def plot_mask(self, image, mask, score, input_point, input_label):
        
        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(coords, labels, ax, marker_size=375):
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Score: {score[0]:.3f}", fontsize=18)
        plt.axis('on')
        plt.show() 
        
        
    def get_binary_mask(self, image_name, plot=False, background=[10,10], foreground=None):
        image = cv2.imread(image_name)
        dimensions = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.generator.set_image(image)
        if foreground==None:
            foreground=[dimensions[1]/2, dimensions[0]/2]
        input_point = np.array([background, foreground])
        
        # 0 denotes background point, 1 denotes foreground point
        input_label = np.array([0,1])
        
        mask, scores, logits = self.generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False, #only generates a single mask (to identify a single article of clothing)
        )
        
        if plot==True:
            self.plot_mask(image, mask, scores, input_point, input_label)
        
        # mask is a H x W binary array
        # the predictor model chooses the mask with the highest score
        return mask, scores, logits
    
    
    
   
##EXAMPLE 
masker = TrainMasker()
mask, score, logit = masker.get_binary_mask(image_name='10044.jpg', plot=True)

## EXAMPLE 2:
mask, score, logit = masker.get_binary_mask(image_name='10011.jpg', plot=True, foreground=[700,400])