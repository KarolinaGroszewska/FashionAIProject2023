# manual spot-checks:

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from your_classification_module import ClothingClassifier  # replace with your actual classifier module

# load the classifier
classifier = ClothingClassifier()

# randomly select 10 images from the dataset for spot-checking
num_images_to_check = 10
image_paths = random.sample(list_of_image_paths, num_images_to_check)

# spot-check each image
for image_path in image_paths:
    # load the image
    image = Image.open(image_path)

    # classify the image
    predicted_class = classifier.classify_image(image)

    # plot the image and its predicted class
    plt.imshow(image)
    plt.title(f"Predicted class: {predicted_class}")
    plt.show()
