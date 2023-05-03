"""
Assumptions:

list_of_image_paths: list of file paths to images in the dataset
list_of_true_labels: list of true labels corresponding to the images in the dataset
ClothingClassifier class: class with a classify_image method that takes in an image and returns a predicted label
For comparing the performance of the model to other state-of-the-art models, benchmark datasets such as Fashion-MNIST or the DeepFashion dataset can be used. The model can be trained on the same dataset as these benchmarks and compared against the reported performance of other models.
"""

from sklearn.metrics import precision_recall_fscore_support

# load the classifier
classifier = ClothingClassifier()

# create empty lists to store predicted and true labels
predicted_labels = []
true_labels = []

# loop through all images in the dataset
for image_path, true_label in zip(list_of_image_paths, list_of_true_labels):
    # load the image
    image = Image.open(image_path)

    # classify the image
    predicted_label = classifier.classify_image(image)

    # store the predicted and true labels
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

# calculate precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
