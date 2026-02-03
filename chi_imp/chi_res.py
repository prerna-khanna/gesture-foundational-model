# -*- coding: utf-8 -*-
"""CHI multiview CNN-geometric property CNN model with stratified split

Modified to use 10% stratified split instead of leave-one-user-out
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import keras
import os
import glob
import cv2
from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
# Set up GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU is available: {physical_devices}")
else:
    print("GPU is NOT AVAILABLE, using CPU instead")

"""# Time series/ feature data preparation"""

def get_features(arr):
    min_ = np.amin(arr)
    max_ = np.amax(arr)
    range_ = max_-min_
    mean_val = np.mean(arr)
    med = np.median(arr)
    std_dev = np.std(arr)
    cv = (std_dev/mean_val)*100
    zcr = np.sum(np.abs(np.diff(arr > np.mean(arr))))/len(arr)
    sk = skew(arr)
    k = kurtosis(arr)
    area = np.trapz(arr)
    energy = np.sqrt(np.mean(np.square(arr)))
    quant = np.quantile(arr,0.5)

    peaks, _ = find_peaks(arr, height=0.04, distance=20)
    peak_width_full = peak_widths(arr, peaks, rel_height=1)
    width = peak_width_full[0]

    fft_ = np.fft.fft(arr)
    fft_ = fft_[:10].tolist()

    ff = []
    for i in fft_:
      fft_ = (str(i))[1:-1]
      ff.append(complex(fft_))

    feat = [min_, max_, range_, mean_val, med, std_dev, cv,
            zcr, sk, k, area, energy, quant,
            len(peaks),
            #peak_width_full[0]
            ]
    width = list(width)
    for pad in range(10-len(width)):
      width.append(0)
    for i in width:
      feat.append(i)

    for i in ff:
      feat.append(i)

    return feat

# Load feature data
feat = pd.read_csv('chi_imp/features_full.csv',  header= None)
targets = pd.read_csv('chi_imp/gesture_full.csv', header= None)
y = targets.iloc[:,0]
user_id = targets.iloc[:,1]

print(y.shape)
total = pd.concat([feat, y], axis=1)
total = total.fillna(total.mean())

n = 180

# Process x trajectory features
x_traj_arr = total.iloc[:,:n]
x_traj = []
for i in range(len(x_traj_arr)):
  x_traj.append(get_features(x_traj_arr.iloc[i,:]))
x_traj = np.array(x_traj[:])

# Process y trajectory features
y_traj_arr = total.iloc[:,n:2*n]
y_traj = []
for i in range(len(y_traj_arr)):
  y_traj.append(get_features(y_traj_arr.iloc[i,:]))
y_traj = np.array(y_traj[:])

# Process z trajectory features
z_traj_arr = total.iloc[:,2*n:3*n]
z_traj = []
for i in range(len(z_traj_arr)):
  z_traj.append(get_features(z_traj_arr.iloc[i,:]))
z_traj = np.array(z_traj[:])

# Process curvature features
curv_arr = total.iloc[:,3*n:4*n]
curv = []
for i in range(len(curv_arr)):
  curv.append(get_features(curv_arr.iloc[i,:]))
curv = np.array(curv[:])

# Process torsion features
tor_arr = total.iloc[:,4*n:5*n]
tor = []
for i in range(len(tor_arr)):
  tor.append(get_features(tor_arr.iloc[i,:]))
tor = np.array(tor[:])

# Process CDF features
cdf_arr = total.iloc[:,5*n:-1]
cdf = []
for i in range(len(cdf_arr)):
  cdf.append(get_features(cdf_arr.iloc[i,:]))
cdf = np.array(cdf[:])

y = total.iloc[:,-1]
y = np.array(y.iloc[:])

# Combine all features
x = np.concatenate((x_traj, y_traj, z_traj, curv, cdf, tor), axis=1)

# Handle complex numbers and ensure all values are numpy arrays (not tensors)
ans = np.array([x.real, x.imag])
arr = np.dstack((ans[0], ans[1]))
arr1 = arr.reshape(arr.shape[0], arr.shape[1]*2)
arr1 = preprocessing.scale(arr1, with_mean=True)
x = np.array(arr1, dtype=np.float32)  # Ensure numpy array type

print(x.shape)
print(np.unique(y, return_counts=True))

"""# Isomer data preprocess"""

# Process image data
proc_data_dir = 'chi_imp/processed_full'
filenames = (glob.glob(os.path.join(proc_data_dir, '*.png')))

# Define a custom sorting function
def custom_sort(filename):
    parts = os.path.basename(filename).split('_')
    label = int(parts[0])
    user = int(parts[1])
    isomer = int(parts[2].split('.')[0])
    return label, user, isomer

# Sort the file names
filenames = sorted(filenames, key=custom_sort)

result_dict = {}

for fname in filenames:
    img_name = fname.split('/')[-1]
    src = cv2.imread(fname)

    gesture = img_name.split('_')[0]
    user = img_name.split('_')[1]
    isomer = (img_name.split('_')[-1]).split('.')[0]

    if gesture not in result_dict:
        result_dict[gesture] = {}
    if user not in result_dict[gesture]:
        result_dict[gesture][user] = []
    result_dict[gesture][user].append(src)

# Convert the result_dict dictionary to a Pandas DataFrame
df = pd.DataFrame.from_dict({(gesture, user): isomers
                             for gesture, users in result_dict.items()
                             for user, isomers in users.items()},
                            orient='index',
                            columns=['isomer1', 'isomer2', 'isomer3', 'isomer4', 'isomer5', 'isomer6'])

df = df.reset_index()
y_img = df['index'][:]
df.head()

a = np.array(df.iloc[:,1:])
x_img = []

for i in range(len(a)):
  for j in range(6):
    x_img.append(a[i][j])

print(np.shape(x_img))

x_img = np.reshape(x_img, (len(df), 6, 128, 128, 3))

y_int = []
for i in range(len(y_img)):
  y_int.append(int(y_img[i][0]))
y_int = np.array(y_int[:])

print(np.unique(y_int, return_counts=True))

"""# Use 10% Stratified Split instead of Leave One User Out"""

# Perform stratified 10% train-test split
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=42, stratify=y
)

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    x_img, y_int, test_size=test_size, random_state=42, stratify=y_int
)

# Print class distribution to verify stratification
print("Training class distribution:", np.unique(y_train, return_counts=True))
print("Testing class distribution:", np.unique(y_test, return_counts=True))

"""# Build and train the model"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, BatchNormalization, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet, EfficientNetB0, MobileNetV2, ResNet50V2, EfficientNetB6
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

# Set the input shape for the images
input_shape = (128, 128, 3)
feat_tensor = Input(shape=(408, 1))

# Create a base model using EfficientNetB0 as the feature extractor
base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

# Freeze the weights in the base model
for layer in base_model.layers:
    layer.trainable = False

# Define the input tensor for the views
input_tensor = Input(shape=(6,) + input_shape)

# Create a list of feature maps for each view
view_features = []
for i in range(6):
    l = Conv2D(32, (3, 3), activation='relu')(base_model(input_tensor[:,i,:,:,:]))
    l = MaxPooling2D((2, 2))(l)
    l = Flatten()(l)
    view_features.append(l)

# Concatenate the feature maps from each view
if len(view_features) > 1:
    cnn_out = concatenate(view_features)
else:
    cnn_out = view_features[0]

# Add dropout layers to prevent overfitting
cnn_out = Dropout(0.3)(cnn_out)
cnn_out = Dense(248, activation='relu')(cnn_out)

# Build the 1D conv model
feat_out = Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(408,1))(feat_tensor)
feat_out = Conv1D(filters=64, kernel_size=3, activation='relu')(feat_out)
feat_out = Conv1D(filters=32, kernel_size=3, activation='relu')(feat_out)
feat_out = Dropout(0.5)(feat_out)
feat_out = MaxPooling1D(pool_size=2)(feat_out)
feat_out = Flatten()(feat_out)

# Concatenate the output from the CNN and the 1D Conv
combined = concatenate([cnn_out, feat_out])

combined = Dense(128, activation='relu')(combined)
combined = Dense(64, activation='relu')(combined)
combined = Dense(32, activation='relu')(combined)
combined = Dense(16, activation='softmax', kernel_regularizer='l2',
                 bias_regularizer='l2')(combined)

# Create the final model
model = Model(inputs=[input_tensor, feat_tensor], outputs=combined)

# Compile the model with appropriate loss and optimizer functions
loss_fn = SparseCategoricalCrossentropy()
optimizer = Adam(0.0001)

# Define a custom learning rate scheduler
def lr_scheduler(epoch):
    return 0.001 * 0.9**epoch

scheduler = LearningRateScheduler(lr_scheduler)

# Fix for EagerTensor serialization issues
import tensorflow.keras.backend as K
K.clear_session()

model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

# Use a simpler checkpoint callback to avoid JSON serialization issues
checkpoint = ModelCheckpoint(
    'model-best.h5',
    verbose=1,
    monitor='val_accuracy',
    save_best_only=True,
    mode='auto'
)

# Format the input data properly
X_train_reshape = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshape = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to numpy arrays
y_train_img_np = np.array(y_train_img, dtype=np.int32)
y_test_img_np = np.array(y_test_img, dtype=np.int32)

# Train the model with proper data formatting
from tensorflow.keras.callbacks import EarlyStopping

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    [X_train_img, X_train_reshape], 
    y_train_img_np, 
    epochs=100,
    batch_size=16,  # Add batch size to reduce memory usage
    validation_data=([X_test_img, X_test_reshape], y_test_img_np),
    callbacks=[scheduler, checkpoint, early_stopping]
)

"""# Evaluate the model"""

# Predict on test data
y_pred = np.argmax(model.predict([X_test_img, X_test_reshape], verbose=0), axis=1)
print("Classification Report:")
print(classification_report(y_test_img_np, y_pred))

# Calculate accuracy
acc = accuracy_score(y_test_img_np, y_pred)
print('Accuracy: %.3f' % acc)



# save the results as a csv file
df = pd.DataFrame({
    'true_label': y_test_img,
    'predicted_label': y_pred
})

# Save to CSV
base_filename = 'results/chi_res/chi_10.csv'
print("save_file_test_size", base_filename)
df.to_csv(base_filename, index=False)
print(f"Predictions and true labels saved to {base_filename}")
