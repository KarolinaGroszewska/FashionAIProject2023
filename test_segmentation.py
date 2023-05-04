import unittest
import os
import cv2
import numpy as np

from segmentation_train import TrainMasker


class TestSegmentation(unittest.TestCase):

    def test_get_binary_mask(self):
        # Initialize the masker
        masker = TrainMasker()
        
        # Load an example image
        image_path = os.path.join(os.getcwd(), 'example_images', '10044.jpg')
        image = cv2.imread(image_path)
        
        # Get the binary mask
        mask, score, logits = masker.get_binary_mask(image_name=image_path)
        
        # Make sure the output is a binary mask with the same dimensions as the input image
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape[:2], image.shape[:2])
        
        # Make sure the score and logits have the expected shape
        self.assertIsInstance(score, np.ndarray)
        self.assertEqual(score.shape, (1,))
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(logits.shape, (1, 2))
        
        # Test with a different image and foreground point
        image_path = os.path.join(os.getcwd(), 'example_images', '10011.jpg')
        image = cv2.imread(image_path)
        mask, score, logits = masker.get_binary_mask(image_name=image_path, foreground=[700,400])
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape[:2], image.shape[:2])
        self.assertIsInstance(score, np.ndarray)
        self.assertEqual(score.shape, (1,))
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(logits.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
