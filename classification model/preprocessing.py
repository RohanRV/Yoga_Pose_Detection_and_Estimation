#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd 
import os
from movenet import Movenet
import wget
import csv
import tqdm 
from data import BodyPart


# In[1]:


if('movenet_thunder.tflite' not in os.listdir()):
    wget.download('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite', 'movenet_thunder.tflite')

movenet = Movenet('movenet_thunder')


# In[2]:


# Define functions to run pose Estimation
def detection_on_input_image(input_tensor, inference_count=3):
    """
    Runs detection on an input image.

    Args:
      input_tensor: A [height, width, 3] tensor of type tf.float32.
    
      inference_count: Number of times the model should run repeatedly
          on the same input image to improve detection accuracy.
      
    Returns:
      A person entity detected by the MoveNet.SinglePose
    """
    image_height, image_width, channel = input_tensor.shape

    # Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    #To identify the region of interest and crop that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(),
                                reset_crop_region=False)
    
    return person


# In[3]:


# Load images, detect pose landmarks and save to CSV 
class MoveNet_Preprocessor(object):
    """Helper class to preprocess pose sample images for classification."""
  
    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_path):
        """Creates a preprocessor to detect pose from images and save as csv.

        Args:
          images_in_folder: Path to folder with input images. 
          images_out_folder: Path to write the images overlay with detected
          landmarks. 
          csvs_out_path: Path to write the CSV containing the detected landmark
          coordinates and label of each image.
        """
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

        # Create a temporary directory to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # Get list of pose classes and print image statistics 
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if not \
             n.startswith('.')])
    
    def preprocess(self, per_pose_class_limit=None, detection_threshold=0.1):
        """Preprocesses images in the given folder.
        Args:
          per_pose_class_limit: No. images to load.
          detection_threshold: Only keep images with all landmark confidence
              score above this threshold
        """
        # Loop through the classes and preprocess its images
        for pose_class_name in self._pose_class_names:
            print('Preprocessing', pose_class_name, file=sys.stderr)

            # Paths for the pose class
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class, 
                                        pose_class_name + '.csv')
      
            # if path does not exist then create the images_out_folder
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            # Detect landmarks and write it to a CSV file
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, 
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL)
                # Get list of images
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                valid_image_count = 0

                # Detect pose landmarks from each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._messages.append('Skipped ' + image_path + '. Invalid image.')
                        continue
                    else:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                        image_height, image_width, channel = image.shape

                    # Skip images other than RGB.
                    if channel != 3:
                        self._messages.append('Skipped ' + image_path + 
                                              '. Image is not in the RGB format.')
                        continue
                    person = detection_on_input_image(image)

                    # Save landmarks if all landmarks were detected 
                    min_landmarks_score = min(
                        [keypoint.score for keypoint in person.keypoints])
                    should_keep_image = min_landmarks_score >= detection_threshold
                    if not should_keep_image:
                        self._messages.append('Skipped ' + image_path + 
                                              '. No pose was confidentally detected.')
                        continue
          
                    valid_image_count += 1

                    # Draw the prediction result on top of the image
                    output_overlay = draw_prediction_on_image(
                        image.numpy().astype(np.uint8), person, 
                        close_figure=True, keep_input_size=True)
          
                    # Write detection result into an image file
                    output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # Scale landmarks to the same size as the input image
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                        for keypoint in person.keypoints],
                        dtype=np.float32)
          
                    # Write landmark coordinates to its per-class CSV file
                    landmark_coordinates = pose_landmarks.flatten().astype(str).tolist()
                    csv_out_writer.writerow([image_name] + landmark_coordinates)

                if not valid_image_count:
                    raise RuntimeError(
                        'No valid images for the "{}" class.'
                        .format(pose_class_name))
    
        # Print the error message collected during preprocessing 
        print('\n'.join(self._messages))

        # Combine all per-class CSVs into a single output file
        all_landmarks_df = self._all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)
    
    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def _all_landmarks_as_dataframe(self):
        """Merge all per-class CSVs into a single dataframe."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        class_name + '.csv')
            per_class_df = pd.read_csv(csv_out_path, header=None)

            # Add labels to the dataframe
            per_class_df['class_no'] = [class_index]*len(per_class_df)
            per_class_df['class_name'] = [class_name]*len(per_class_df)

            # Append the folder name to the filename column
            per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '')
                                                     + per_class_df[per_class_df.columns[0]].astype(str))

            if total_df is None:
                # For the first class, assign its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate each class's data into the total dataframe
                total_df = pd.concat([total_df, per_class_df], axis=0)

        list_name = [[bodypart.name + '_x', bodypart.name + '_y', 
                      bodypart.name + '_score'] for bodypart in BodyPart]

        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ['file_name'] + header_name 
        header_map = {total_df.columns[i]: header_name[i]
                      for i in range(len(header_name))}

        total_df.rename(header_map, axis=1, inplace=True)

        return total_df     


# ### Preprocess the TRAIN dataset

# In[5]:


images_in_train_folder = "/content/yoga_poses/train/"
images_out_train_folder = 'poses_images_out_train'
csvs_out_train_path = 'train_data.csv'

preprocessor = MoveNet_Preprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
)

preprocessor.preprocess(per_pose_class_limit=None)


# ### Preprocess the TEST dataset

# In[6]:


images_in_test_folder = "/content/yoga_poses/test/"
images_out_test_folder = 'poses_images_out_test'
csvs_out_test_path = 'test_data.csv'

preprocessor = MoveNet_Preprocessor(
    images_in_folder=images_in_test_folder,
    images_out_folder=images_out_test_folder,
    csvs_out_path=csvs_out_test_path,
)

preprocessor.preprocess(per_pose_class_limit=None)

