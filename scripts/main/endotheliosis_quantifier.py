import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
from typing import List


# my annotation class object
class Annotation:
    def __init__(self, image_name, rle_mask, score=None):
        self.image_name = image_name
        self.rle_mask = rle_mask
        self.score = score

    def __repr__(self):
        return f"Annotation(image_path={self.image_name}, annotations={self.rle_mask}, score={self.score})"

# my rle input stream class
class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)

# Display images
def display_image(img, window_name):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Decode the RLE semantic segmentation mask into a binary mask
def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])

def rle_to_binary_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image

# extract the annotations from the json file produced by label-studio
def load_annotations_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)

    annotations = []
    for entry in annotations_data:
        image_name = entry['file_upload']
        # print(image_name)
        image_name = image_name.split('-')[1]
        # print(image_path)

        image_rle_mask = []
        score = None
        for annotation in entry['annotations']:
            # Find 'rle' values
            rle_values = [result['value']['rle'] for result in annotation['result'] if 'rle' in result['value']]
            if rle_values:
                image_rle_mask.extend(rle_values[0])

            # Find 'choices' values and extract the score
            choices_values = [result['value']['choices'] for result in annotation['result'] if 'choices' in result['value']]
            if choices_values and score is None:
                score = float(choices_values[0][0])
        
        annotation = Annotation(image_name, image_rle_mask, score)
        annotations.append(annotation)
        print(annotation.image_name, annotation.score)
        
    return annotations

# preprocess the image a bit
def preprocess_image(image_path):
    # Load the image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # display_image(img, "Original Grayscale Image")
    
    # Apply Gaussian blur
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # display_image(img_blurred, "Blurred Image")
    
    return img_blurred

# get the image path from the json file of annotations
def get_image_path_from_json(json_file):
    with open(json_file, 'r') as f:
        annotations_data = json.load(f)
    return [entry['file_upload'] for entry in annotations_data]

# extract the score from each annotation
def get_scores_from_annotations(annotations):
    labels = []
    for annotation in annotations:
        if annotation.score is not None:
            label = int(annotation.score)
        else:
            label = None
        labels.append(label)
    return labels

# Match up the name of the image that was uploaded to label
# and the image name you have in your data directory
def find_image_path(image_name, root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == image_name:
                return os.path.join(root, file)
    return None

def openness_score(glomerulus_contour, preprocessed_image):
    # Create a binary mask with the same dimensions as the input image
    mask = np.zeros_like(preprocessed_image, dtype=np.uint8)

    # Fill the glomerulus contour with white color
    cv2.drawContours(mask, [glomerulus_contour], 0, 255, -1)

    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)

    # Calculate the area of open capillaries (white pixels in the preprocessed image within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(preprocessed_image, mask))

    # Calculate the ratio of open area to total area
    score = open_area / total_area if total_area > 0 else 0

    return score

def grade_glomerulus(openness_score):
    # Define the threshold values for each grade based on your ground-truth data
    grade_thresholds = [0.6, 0.4, 0.2] # 20% open, 40% open, 60% open

    # Grade the glomerulus based on the openness score
    for i, threshold in enumerate(grade_thresholds):
        if openness_score >= threshold:
            return i
    return len(grade_thresholds)


# this doesn't work super well...
def segment_glomeruli_and_binary_mask(preprocessed_image):
    # # Apply Otsu's thresholding method
    # _, binary_image = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
     # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological opening and closing operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, img_opened, img_closed 

# extract the contour features
def extract_features(contours):
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # # Calculate the openness score for each contour using the preprocessed_image
        # openness = openness_score(cnt, preprocessed_image)

        features.append([area, circularity])
    return np.array(features)


# Load all the data up
def load_data(annotation_file, data_dir):
    """Load data, convert rle mask to 

    Args:
        annotation_file (_type_): _description_
        data_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = {}
    
    # Load the annotations
    annotations = load_annotations_from_json(annotation_file)

    for annotation in annotations:
        img_name = annotation.image_name
        img_path = find_image_path(img_name, root_directory = data_dir)
        
        if img_path is None:
            print(f"Image file {img_name} not found in the directory.")
            continue

        # Load the image and its dimensions
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Decode the RLE mask to a binary image
        binary_mask = rle_to_binary_mask(annotation.rle_mask, img_height, img_width)
        # display_image(binary_mask, "Binary Mask")
        
        # Preprocess the image and find contours from the binary mask
        # preprocessed_image = preprocess_image(img_path)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract features from the contours
        features = extract_features(contours)  # Pass the preprocessed_image
        # Access the ordinal score for the glomerulus, from training data
        score = annotation.score

        # Ensure that score is a float: [0.0, 0.5, 1.0, 1.5, etc]
        if score is not None:
            score = float(score)

        # Add the features and score to the dictionary
        if img_name not in data:
            data[img_name] = {'X': [], 'y': []}
        data[img_name]['X'].extend(features)
        data[img_name]['y'].extend([score] * len(features))

    return data

def trainModel(annotation_file, data_dir):
    """Train the regression model 

    Args:
        annotation_file (str): annotation file exported from label-studio, containing score and rle
        data_dir (str): denotes the data directory with all the raw images
    """
    print('Loading the data...')
    data = load_data(annotation_file, data_dir)

    # Get the image names
    image_names = list(data.keys())

    print('Training model...')
    # Split the image names into training and testing sets
    train_image_names, test_image_names = train_test_split(image_names, test_size=0.2, random_state=42)

    # Get the X and y arrays for the training and testing sets
    X_train = []
    y_train = []
    for image_name in train_image_names:
        X_train.extend(data[image_name]['X'])
        y_train.extend(data[image_name]['y'])
    # Convert to numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for image_name in test_image_names:
        X_test.extend(data[image_name]['X'])
        y_test.extend(data[image_name]['y'])
    # Convert to numpy
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # print(f'X_train: {X_train.shape}')
    # print(f'X_test: {X_test.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'y_test: {y_test.shape}')

    # Print the names of the images in the training and testing sets
    print('Training images:', train_image_names)
    print('Testing images:', test_image_names)

    # # Train the classifier
    endotheliosisQuantifierModel = RandomForestRegressor(n_estimators=100)
    endotheliosisQuantifierModel.fit(X_train, y_train)

    print('Calculating endotheliosis on test set...')
    # # Predict the labels for the test set
    y_pred = endotheliosisQuantifierModel.predict(X_test)

    print('Evaluating performance...')
    # Evaluate the performance
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    return(endotheliosisQuantifierModel)



def inspect_images(original_image_path, binary_mask):
    #  preprocessed_image,
    original_image = cv2.imread(original_image_path)

    # Concatenate the images horizontally
    combined_image = np.concatenate((original_image, 
                                    #  cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR), 
                                     cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)), axis=1)
    
    # Display the concatenated image
    display_image(combined_image, "Original and Binary Mask")

def process_new_image(image_path, regressor):
    
    print(f'Processing new image {image_path}...')
    preprocessed_image = preprocess_image(image_path)

    # Use the regressor to predict the scores for each glomerulus
    contours, _, binary_mask = segment_glomeruli_and_binary_mask(preprocessed_image)
    # contours, binary_mask = segment_glomeruli_and_binary_mask_watershed(preprocessed_image)
    # inspect_images(image_path, binary_mask)

    # features = extract_features(contours)
    # scores = regressor.predict(features)

    # Find contours in the binary mask
    mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Match each contour in the binary mask with the closest contour found by segment_glomeruli_and_binary_mask
    all_grades = []
    all_scores = []
     # Add the features and score to the dictionary
    for mask_cnt in mask_contours:
        min_distance = float('inf')
        matched_cnt = None
        for cnt in contours:
            distance = cv2.matchShapes(cnt, mask_cnt, cv2.CONTOURS_MATCH_I1, 0)
            if distance < min_distance:
                min_distance = distance
                matched_cnt = cnt

        # Calculate the glomerulus grade for the matched contour
        if matched_cnt is not None:
            matched_features = extract_features([matched_cnt])
            score = regressor.predict(matched_features)[0]
            grade = grade_glomerulus(score)

            # print(f'Contour grade: {grade}')
            all_grades.append(grade)
            all_scores.append(score)
            
    result_dict = {}
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dict[image_name] = {'Grades': {'mean_grade': np.mean(np.array(all_grades)),
                                          'median_grade': np.median(np.array(all_grades))}, 
                                'Scores': {'mean_score': np.mean(np.array(all_scores)),
                                           'median_score': np.median(np.array(all_scores))}
                                }
    return(result_dict)




annotation_file = 'annotations.json'
data_dir = 'jpg_data'

endotheliosisQuantifierModel = trainModel(annotation_file, data_dir)

# Test on one image
image_path = 'jpg_data/Lauren_PreEclampsia_Raw_Images/T30/T30_Image0.jpg'
result = process_new_image(image_path, endotheliosisQuantifierModel)
print("Results:")
print(json.dumps(result, indent=2))


## RUN THIS IN A LOOP
# images_dir = 'jpg_data/Lauren_PreEclampsia_Raw_Images'
# output_dir = 'output'

# # create an empty dataframe to store the results
# results_df = pd.DataFrame(columns=['subdirectory', 'image', 'result'])

# # loop through subdirectories and files
# for subdir, dirs, files in os.walk(images_dir):
#     print(subdir)
#     for file in files:
#         # check if file is a .jpg
#         if file.endswith('.jpg') or file.endswith('.jpeg'):
#             # construct the full path to the image
#             image_path = os.path.join(subdir, file)
#             # process the image and store the result
#             result = process_new_image(image_path, endotheliosisQuantifierModel)
#             print(result)
#             # add the result to the dataframe
#             results_df = results_df.append({'subdirectory': subdir,
#                                             'image': file,
#                                             'result': result},
#                                            ignore_index=True)

# # save the results to a CSV file
# results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

# # group the results by subdirectory and image, and calculate the mean result
# avg_results_df = results_df.groupby(['subdirectory', 'image']).mean().reset_index()

# # save the average results to a CSV file
# avg_results_df.to_csv(os.path.join(output_dir, 'average_results.csv'), index=False)  
      

# print("Done!")