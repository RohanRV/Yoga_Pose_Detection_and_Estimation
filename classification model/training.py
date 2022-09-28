#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import csv
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from data import BodyPart 
import tensorflow as tf
import tensorflowjs as tfjs


# In[ ]:


tfjs_model_dir = 'model'


# In[ ]:


def load_pose_landmarks(csv_path):
    """Loads a CSV created by MoveNetPreprocessor.
  
    Returns:
      X: Detected landmark coordinates and scores 
      y: Ground truth labels 
      classes: The list of all class names found in the dataset
      dataframe: The CSV loaded as a Pandas dataframe.
    """

    # read the CSV file
    df = pd.read_csv(csv_path)
    df_to_process = df.copy()

    # Drop the file_name columns
    df_to_process.drop(columns=['file_name'], inplace=True)

    # Extract the list of class names
    classes = df_to_process.pop('class_name').unique()

    # Extract the labels
    y = df_to_process.pop('class_no')

    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes, df


# In[ ]:


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of landmarks."""

    left_landmark = tf.gather(landmarks, left_bodypart.value, axis=1)
    right_landmark = tf.gather(landmarks, right_bodypart.value, axis=1)
    center_point = left_landmark * 0.5 + right_landmark * 0.5
    return center_point


# In[ ]:


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size"""
    # Hips center
    hips_center_point = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                         BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center_point = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                              BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center_point - hips_center_point)

    # New Pose center
    new_pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                       BodyPart.RIGHT_HIP)
    new_pose_center = tf.expand_dims(new_pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    new_pose_center= tf.broadcast_to(new_pose_center,
                                     [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    distance = tf.gather(landmarks - new_pose_center, 0, axis=0,
                         name="dist_to_pose_center")
    # Max dist to pose center
    max_distance = tf.reduce_max(tf.linalg.norm(distance, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_distance)

    return pose_size


# In[ ]:


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation"""
    # Move landmarks so that the pose center becomes (0,0)
    pose_center_point = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                         BodyPart.RIGHT_HIP)
    pose_center_point = tf.expand_dims(pose_center_point, axis=1)
    # Broadcast the pose center to the same size 
    pose_center_point = tf.broadcast_to(pose_center_point, 
                                        [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center_point

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


# In[ ]:


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix 
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


# In[ ]:


# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)


# In[ ]:


# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.15)


# In[ ]:


# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)


# In[ ]:


# Define the model
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.BatchNormalization()(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.BatchNormalization()(layer)
layer = keras.layers.Dense(32, activation=tf.nn.relu6)(layer)
layer = keras.layers.BatchNormalization()(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
model.summary()


# In[ ]:


from tensorflow import keras
keras.utils.plot_model(model)


# In[ ]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Store the checkpoint that has the highest validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                              patience=30)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, earlystopping])


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')
plt.show()


# In[ ]:


# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(X_test, y_test)


# ### Plot the confusion matrix

# In[ ]:


def confusion_matrix_plot(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
confusion_matrix_plot(cm,
                      class_names,
                      title ='Confusion Matrix of Pose Classification Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                          y_pred_label))


# ### Investigate Incorrect Predictions

# In[ ]:


IMAGE_PER_ROW = 3
MAX_NO_OF_IMAGE_TO_PLOT = 30

# Extract the list of incorrectly predicted poses
false_predicted_pose = [id_in_df for id_in_df in range(len(y_test))                 if y_pred_label[id_in_df] != y_true_label[id_in_df]]
if len(false_predicted_pose) > MAX_NO_OF_IMAGE_TO_PLOT:
    false_predicted_pose = false_predicted_pose[:MAX_NO_OF_IMAGE_TO_PLOT]

# Plot the incorrectly predicted images
row_count = len(false_predicted_pose) // IMAGE_PER_ROW + 1
fig = plt.figure(figsize=(10 * IMAGE_PER_ROW, 10 * row_count))
for i, id_in_df in enumerate(false_predicted_pose):
    ax = fig.add_subplot(row_count, IMAGE_PER_ROW, i + 1)
    image_path = os.path.join(images_out_test_folder,
                              df_test.iloc[id_in_df]['file_name'])

    image = cv2.imread(image_path)
    plt.title("Predicted_pose: %s; Actual_pose: %s"
              % (y_pred_label[id_in_df], y_true_label[id_in_df]))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


# ### Investigate Correct Predictions

# In[ ]:


#prediction function 
IMAGE_PER_ROW = 3
MAX_NO_OF_IMAGE_TO_PLOT = 30

# Extract the list of correctly predicted poses
correct_predicted_pose= [id_in_df for id_in_df in range(len(y_test))                          if y_pred_label[id_in_df] == y_true_label[id_in_df]]
if len(correct_predicted_pose) > MAX_NO_OF_IMAGE_TO_PLOT:
    correct_predicted_pose = correct_predicted_pose[:MAX_NO_OF_IMAGE_TO_PLOT]

# Plot the correctly predicted images

row_count = len(correct_predicted_pose) // IMAGE_PER_ROW + 1
fig = plt.figure(figsize=(10 * IMAGE_PER_ROW, 10 * row_count))
for i, id_in_df in enumerate(correct_predicted_pose):
    ax = fig.add_subplot(row_count, IMAGE_PER_ROW, i + 1)
    image_path = os.path.join(images_out_test_folder,
                              df_test.iloc[id_in_df]['file_name'])

    image = cv2.imread(image_path)
    plt.title("Predicted_pose: %s; Actual_pose: %s"
              % (y_pred_label[id_in_df], y_true_label[id_in_df]))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

