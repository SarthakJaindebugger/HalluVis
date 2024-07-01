# -*- coding: utf-8 -*-
"""EmbeddingExtraction+Classifiers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F2KYgm-vNEPJiO2T3Tzf-qJOcWm9ReJx

#Embedding Extraction
"""

"""
##VideoMAE Embedding Extraction """

import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import VideoMAEImageProcessor, VideoMAEModel
import pickle

# Initialize VideoMAE processor and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
    indices = np.clip(indices, 0, seg_len - 1)
    return indices

def read_video_moviepy(video_clip, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    video_clip.reader.close()
    return np.stack(frames)

def extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels):
    embeddings = []
    multi_labels = []
    for video_path, label in zip(video_paths, labels):
        video_clip = VideoFileClip(video_path)
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=int(video_clip.duration * video_clip.fps))
        video_frames = read_video_moviepy(video_clip, indices)
        inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = video_model(**inputs)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings.append(last_hidden_states.cpu().numpy())

        # Convert labels to multi-label binary vector
        label_vector = np.zeros(num_labels)
        for l in label:
            label_vector[l] = 1
        multi_labels.append(label_vector)

    with open(output_embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    with open(output_labels_file, 'wb') as f:
        pickle.dump(multi_labels, f)

# Example usage
video_paths = [
    "/EmbeddingAndClassifiers/videos/2 cats who are playing outside of a house.mp4",
    "/EmbeddingAndClassifiers/videos/2 men on a court play a game of tennis.mp4",
    "/EmbeddingAndClassifiers/videos/2 people pose for a picture by a cold mountain.mp4",
    "/EmbeddingAndClassifiers/videos/2 teddy bears in matching clothing next to each other.mp4",
    "/EmbeddingAndClassifiers/videos/3 people smile in the airport amongst a crowd.mp4"
]  # Add all your video paths here
labels = [
    [3],
    [4, 6],
    [4],
    [3],
    [1, 3, 4, 6]
]  # Add corresponding multi-labels for each video
num_labels = 7  # Total number of unique labels: 0 - Vanishing Subject; 1 - Subject Multiplication/Reduction; 2 - Incongruity Fusion; 3 - Omission Error; 4 - Temporal Subject Dysmorphia; 5 - Action Inference; 6 - Unnatural Physics;

output_embeddings_file = "/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl"
output_labels_file = "/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl"
extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels)

#########################################################################

"""##TimeSformer Embedding Extraction """

import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import AutoImageProcessor, TimesformerModel
import pickle

# Initialize TimeSformer processor and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
video_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
    indices = np.clip(indices, 0, seg_len - 1)
    return indices

def read_video_moviepy(video_clip, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    video_clip.reader.close()
    return np.stack(frames)

def extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels):
    embeddings = []
    multi_labels = []
    for video_path, label in zip(video_paths, labels):
        video_clip = VideoFileClip(video_path)
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=int(video_clip.duration * video_clip.fps))
        video_frames = read_video_moviepy(video_clip, indices)
        inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = video_model(**inputs)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings.append(last_hidden_states.cpu().numpy())

        # Convert labels to multi-label binary vector
        label_vector = np.zeros(num_labels)
        for l in label:
            label_vector[l] = 1
        multi_labels.append(label_vector)

    with open(output_embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    with open(output_labels_file, 'wb') as f:
        pickle.dump(multi_labels, f)

# Example usage
video_paths = [
    "/EmbeddingAndClassifiers/videos/2 cats who are playing outside of a house.mp4",
    "/EmbeddingAndClassifiers/videos/2 men on a court play a game of tennis.mp4",
    "/EmbeddingAndClassifiers/videos/2 people pose for a picture by a cold mountain.mp4",
    "/EmbeddingAndClassifiers/videos/2 teddy bears in matching clothing next to each other.mp4",
    "/EmbeddingAndClassifiers/videos/3 people smile in the airport amongst a crowd.mp4"
]   # Add all your video paths here
labels = [
    [3],
    [4, 6],
    [4],
    [3],
    [1, 3, 4, 6]
]  # Add corresponding multi-labels for each video
num_labels = 7  # Total number of unique labels: 0 - Vanishing Subject; 1 - Subject Multiplication/Reduction; 2 - Incongruity Fusion; 3 - Omission Error; 4 - Temporal Subject Dysmorphia; 5 - Action Inference; 6 - Unnatural Physics;

output_embeddings_file = "/EmbeddingAndClassifiers/embeddings/timesformer_video_embeddings.pkl"
output_labels_file = "/EmbeddingAndClassifiers/embeddings/timesformer_video_labels.pkl"
extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels)


#########################################################################


"""##ViViT Embedding Extraction
Working for longer videos: Luman and Runway.
Not working for Mora videos - possible explanation would be that from smaller videos/GIFs 3137 number of positional embeddings is not being learnt/retrieved, hence this is not matching with ViViT architecture requirement. Workaround: fine-tune the ViViT on smaller videos i.e, low resolution images so that positional embdedding count can be decreased to 1568/1569.
"""

import numpy as np
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import VivitImageProcessor, VivitModel
import pickle

# Initialize ViViT processor and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
video_model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
    indices = np.clip(indices, 0, seg_len - 1)
    return indices

def read_video_moviepy(video_clip, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            resized_frame = Image.fromarray(frame).resize((224, 224))
            frames.append(resized_frame)
    video_clip.reader.close()
    return frames

def extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels):
    embeddings = []
    multi_labels = []
    for video_path, label in zip(video_paths, labels):
        video_clip = VideoFileClip(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=2, seg_len=int(video_clip.duration * video_clip.fps))
        video_frames = read_video_moviepy(video_clip, indices)

        # Ensure frames are correctly processed and resized to 224x224
        inputs = image_processor(images=video_frames, return_tensors="pt").to(device)

        # Print the shape of the inputs before passing to the model
        print(f"Input shape: {inputs['pixel_values'].shape}")

        with torch.no_grad():
            outputs = video_model(**inputs)

        # Print the shape of the outputs after inference
        print(f"Output shape: {outputs.last_hidden_state.shape}")

        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings.append(last_hidden_states.cpu().numpy())

        # Convert labels to multi-label binary vector
        label_vector = np.zeros(num_labels)
        for l in label:
            label_vector[l] = 1
        multi_labels.append(label_vector)

    with open(output_embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    with open(output_labels_file, 'wb') as f:
        pickle.dump(multi_labels, f)

# Example usage
video_paths = [
    # "/EmbeddingsAndClassifiers/videos/2 cats who are playing outside of a house.mp4",
    "/EmbeddingsAndClassifiers/videos/A girl smiles as she holds a cat and wears a brightly colored skirt.mp4",
    "/EmbeddingsAndClassifiers/videos/A baby bird that is sitting in a nest.mp4"
]  # Add all your video paths here
labels = [
    [3],
    [4, 6],
    [4],
    [3],
    [1, 3, 4, 6]
]  # Add corresponding multi-labels for each video
num_labels = 7  # Total number of unique labels: 0 - Vanishing Subject; 1 - Subject Multiplication/Reduction; 2 - Incongruity Fusion; 3 - Omission Error; 4 - Temporal Subject Dysmorphia; 5 - Action Inference; 6 - Unnatural Physics;

output_embeddings_file = "/EmbeddingAndClassifiers/embeddings/vivit_video_embeddings.pkl"
output_labels_file = "/EmbeddingAndClassifiers/embeddings/vivit_video_labels.pkl"
extract_and_save_embeddings(video_paths, labels, output_embeddings_file, output_labels_file, num_labels)


#########################################################################


#########################################################################


"""#Classifiers

##CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the embeddings and multi-labels from pickle files
with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl', 'rb') as f:                          #REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    X = pickle.load(f)

with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl', 'rb') as f:                              #REPLACE LABELS FILE THAT NEEDS TO BE USED
    y = pickle.load(f)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing datasets
X_train, X_test = X[:3], X[3:]
y_train, y_test = y[:3], y[3:]

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_labels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * (input_size - 4), 64)  # Adjust the size according to your input size and kernel sizes
        self.fc2 = nn.Linear(64, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
num_labels = y_train.shape[1]
model = SimpleCNN(input_size=input_size, num_labels=num_labels).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device))
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {accuracy}')

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')

# Generate classification report
class_report = classification_report(y_test, y_pred, digits=6)
print('\nClassification Report:\n', class_report)

# # Generate and visualize the confusion matrix for multi-labels
# confusion_mat = multilabel_confusion_matrix(y_test, y_pred)
# for i, cm in enumerate(confusion_mat):
#     plt.figure(figsize=(5, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+str(i), str(i)], yticklabels=['Not '+str(i), str(i)])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix for Label {i}')
#     plt.show()

# Print the predicted and actual labels for each video embedding
for i in range(len(X_test)):
    print(f'Video {i+1} Embedding:')
    print(f'Predicted Labels: {y_pred[i]}')
    print(f'Actual Labels: {y_test[i].numpy()}')
    print()
    

#########################################################################

"""##RNN"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Load the embeddings and multi-labels from pickle files
with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    X = pickle.load(f)

with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl', 'rb') as f:  # REPLACE LABELS FILE THAT NEEDS TO BE USED
    y = pickle.load(f)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing datasets
X_train, X_test = X[:3], X[3:]
y_train, y_test = y[:3], y[3:]

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data to include the sequence length (1 in this case)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize model, loss function, and optimizer
input_size = X_train.shape[2]
hidden_size = 64
num_labels = y_train.shape[1]
model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, num_labels=num_labels).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 3

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device))
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {accuracy}')

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')

# Generate classification report
class_report = classification_report(y_test, y_pred, digits=6)
print('\nClassification Report:\n', class_report)

# Print the predicted and actual labels for each video embedding
for i in range(len(X_test)):
    print(f'Video {i+1} Embedding:')
    print(f'Predicted Labels: {y_pred[i]}')
    print(f'Actual Labels: {y_test[i].numpy()}')
    print()


#########################################################################

"""##GRU"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the embeddings and multi-labels from pickle files
with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    X = pickle.load(f)

with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl', 'rb') as f:  # REPLACE LABELS FILE THAT NEEDS TO BE USED
    y = pickle.load(f)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing datasets
X_train, X_test = X[:3], X[3:]
y_train, y_test = y[:3], y[3:]

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data to include the sequence length (1 in this case)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the GRU model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize model, loss function, and optimizer
input_size = X_train.shape[2]
hidden_size = 64
num_labels = y_train.shape[1]
model = GRUClassifier(input_size=input_size, hidden_size=hidden_size, num_labels=num_labels).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device))
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {accuracy}')

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')

# Generate classification report
class_report = classification_report(y_test, y_pred, digits=6)
print('\nClassification Report:\n', class_report)

# # Generate and visualize the confusion matrix for multi-labels
# confusion_mat = multilabel_confusion_matrix(y_test, y_pred)
# for i, cm in enumerate(confusion_mat):
#     plt.figure(figsize=(5, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+str(i), str(i)], yticklabels=['Not '+str(i), str(i)])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix for Label {i}')
#     plt.show()

# Print the predicted and actual labels for each video embedding
for i in range(len(X_test)):
    print(f'Video {i+1} Embedding:')
    print(f'Predicted Labels: {y_pred[i]}')
    print(f'Actual Labels: {y_test[i]}')
    print()

#########################################################################

"""##RF"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the embeddings and multi-labels from pickle files
with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    X = pickle.load(f)

with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl', 'rb') as f:  # REPLACE LABELS FILE THAT NEEDS TO BE USED
    y = pickle.load(f)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing datasets
X_train, X_test = X[:3], X[3:]
y_train, y_test = y[:3], y[3:]

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def random_forest_model(X_train, X_test, y_train, y_test):
    # Ensure the number of samples is the same for X and y
    min_samples = min(len(X_train), len(y_train), len(X_test), len(y_test))

    X_train = X_train[:min_samples]
    y_train = y_train[:min_samples]
    X_test = X_test[:min_samples]
    y_test = y_test[:min_samples]

    # Ensure X_train and X_test are 2D arrays
    if len(X_train.shape) == 3:
        # Reshape to 2D (samples, features)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Initialize the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Generate confusion matrix for each label
    confusion_mats = multilabel_confusion_matrix(y_test, y_pred)

    # # Visualize the confusion matrix using Seaborn's heatmap
    # for i, cm in enumerate(confusion_mats):
    #     sns.heatmap(cm, annot=True, fmt='d',
    #                 xticklabels=['Not Label ' + str(i), 'Label ' + str(i)],
    #                 yticklabels=['Not Label ' + str(i), 'Label ' + str(i)])
    #     plt.ylabel('Actual', fontsize=13)
    #     plt.xlabel('Predicted', fontsize=13)
    #     plt.title(f'Confusion Matrix for Label {i}', fontsize=17)
    #     plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Weighted F1 Score: {weighted_f1}')

    # Generate classification report
    class_report = classification_report(y_test, y_pred, digits=6)
    print('Classification Report:\n', class_report)

    # Print the predicted and actual labels for each video embedding
    for i in range(len(X_test)):
        print(f'Video {i+1} Embedding:')
        print(f'Predicted Labels: {y_pred[i]}')
        print(f'Actual Labels: {y_test[i].numpy()}')
        print()

# Example usage
random_forest_model(X_train, X_test, y_train, y_test)


#########################################################################

"""##SVC"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the embeddings and multi-labels from pickle files
with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_embeddings.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    X = pickle.load(f)

with open('/EmbeddingAndClassifiers/embeddings/videoMAE_video_labels.pkl', 'rb') as f:  # REPLACE LABELS FILE THAT NEEDS TO BE USED
    y = pickle.load(f)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing datasets
X_train, X_test = X[:3], X[3:]
y_train, y_test = y[:3], y[3:]

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Support Vector Classifier model
svc = SVC(kernel='linear', probability=True, random_state=42)

# Use OneVsRestClassifier for multi-label classification
model = OneVsRestClassifier(svc)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict the test set
y_pred_prob = model.predict_proba(X_test_scaled)

# Binarize the predictions based on a threshold of 0.5
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')

# Generate classification report
class_report = classification_report(y_test, y_pred, digits=6)
print('Classification Report:\n', class_report)

# # Generate and visualize the confusion matrix for multi-labels
# confusion_mat = multilabel_confusion_matrix(y_test, y_pred)
# for i, cm in enumerate(confusion_mat):
#     plt.figure(figsize=(5, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+str(i), str(i)], yticklabels=['Not '+str(i), str(i)])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix for Label {i}')
#     plt.show()

# Print the predicted and actual labels for each video embedding
for i in range(len(X_test)):
    print(f'Video {i+1} Embedding:')
    print(f'Predicted Labels: {y_pred[i]}')
    print(f'Actual Labels: {y_test[i]}')
    print()
    

#########################################################################