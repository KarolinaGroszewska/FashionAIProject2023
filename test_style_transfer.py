import unittest
import os
import cv2
import numpy as np

from style_transfer_train import TrainStyleTransfer


class TestStyleTransfer(unittest.TestCase):

    def test_transfer_style(self):
        # Initialize the style transferer
        transferer = TrainStyleTransfer()
        
        # Load example content and style images
        content_path = os.path.join(os.getcwd(), 'example_images', 'content.jpg')
        content = cv2.imread(content_path)
        style_path = os.path.join(os.getcwd(), 'example_images', 'style.jpg')
        style = cv2.imread(style_path)
        
        # Transfer the style
        stylized_image = transferer.transfer_style(content, style)
        
        # Make sure the output has the expected shape
        self.assertIsInstance(stylized_image, np.ndarray)
        self.assertEqual(stylized_image.dtype, np.uint8)
        self.assertEqual(stylized_image.shape, content.shape)
        
        # Test with a different style image
        style_path = os.path.join(os.getcwd(), 'example_images', 'style2.jpg')
        style = cv2.imread(style_path)
        stylized_image = transferer.transfer_style(content, style)
        self.assertIsInstance(stylized_image, np.ndarray)
        self.assertEqual(stylized_image.dtype, np.uint8)
        self.assertEqual(stylized_image.shape, content.shape)


if __name__ == '__main__':
    unittest.main()
