
# change this to the path to Masking folder on your laptop
# make sure that test_getMasks.py is under the Masking folder
import sys
sys.path.append('/Users/vanessapei/Documents/GitHub/FashionAIProject2023/Masking')


import numpy as np
import pandas as pd
from tqdm import tqdm
from train_masker import TrainMasker



# 1. Place train_masker.py in the same folder as getMasks.py
# 2. Make sure that the pixel_location csv file contains headers "image", "x", "y", and "location"
# 3. Ensure that the folder of images is also located in this same folder
#    3b. make sure that locations in pixel_locations csv file match file names and hierarchy


# file name to store binary mask data
npz_filename = "mask_data.npz"

# Ask the user to input the name of the pixel location file
# please ucomment this line on your laptop:
# pixel_locations = input("Enter the name of the pixel location file: ")

# please comment this line on your laptop:
pixel_locations = "/Users/vanessapei/Documents/GitHub/FashionAIProject2023/Masking/pixel_locations_actual_VANESSA.csv"
data = pd.read_csv(pixel_locations)


all_masks = []
all_keys = []
masker = TrainMasker()

data = pd.read_csv(pixel_locations)


# For testing purposes (with a VERY SMALL BATCH OF DATA), uncomment plot line to view mask
for i in tqdm(range(data.shape[0])):
    mask, score, logit = masker.get_binary_mask(image_name=data.loc[i]['location'],
                                                # plot=True,
                                                foreground=[data.loc[i]['x'], data.loc[i]['y']])
    all_masks.append(mask[0])
    all_keys.append(str(data.loc[i]['image']))


# Output is stored in a npz file, with each binary mask indexed by (string version) of image index
res = dict(zip(all_keys, all_masks))
np.savez(npz_filename, **res)

# For example, to retrieve data for image 3909.jpg:
#    data = np.load(npz_filename)
#    data['3909']


# Additional test cases
def test_file_exists():
    assert npz_filename in os.listdir(), f"File {npz_filename} does not exist."

def test_data_length():
    assert len(all_masks) == data.shape[0], f"Number of masks does not match number of images."

def test_mask_shape():
    for mask in all_masks:
        assert mask.shape == (256, 256), f"Mask has incorrect shape: {mask.shape}"

def test_mask_range():
    for mask in all_masks:
        assert np.all(np.isin(mask, [0, 1])), f"Mask contains values outside of 0 and 1: {mask}"

def test_retrieval():
    data = np.load(npz_filename)
    for key in all_keys:
        assert key in data.files, f"Key {key} not found in saved data."
        assert data[key].shape == (256, 256), f"Mask for key {key} has incorrect shape: {data[key].shape}"
        assert np.all(np.isin(data[key], [0, 1])), f"Mask for key {key} contains values outside of 0 and 1: {data[key]}"

if __name__ == "__main__":
    test_file_exists()
    test_data_length()
    test_mask_shape()
    test_mask_range()
    test_retrieval()
    print("All tests passed.")
