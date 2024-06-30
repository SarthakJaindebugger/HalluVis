import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Trim the larger dataset to match the size of the smaller one for training set
min_samples_train_lfcc_wavlm = min(X_train_lfcc.shape[0], X_train_wavlm.shape[0])
X_train_combined_lfcc_wavlm = np.concatenate((X_train_lfcc[:min_samples_train_lfcc_wavlm, :], X_train_wavlm[:min_samples_train_lfcc_wavlm, :]), axis=1)
y_train_combined_lfcc_wavlm = y_train[:min_samples_train_lfcc_wavlm]

# Display the shape of the combined training set
print("X_train_combined_lfcc_wavlm shape:", X_train_combined_lfcc_wavlm.shape)
print("y_train_combined_lfcc_wavlm shape:", y_train_combined_lfcc_wavlm.shape)

# Trim the larger dataset to match the size of the smaller one for test set
min_samples_test_lfcc_wavlm = min(X_test_lfcc.shape[0], X_test_wavlm.shape[0])
X_test_combined_lfcc_wavlm = np.concatenate((X_test_lfcc[:min_samples_test_lfcc_wavlm, :], X_test_wavlm[:min_samples_test_lfcc_wavlm, :]), axis=1)
y_test_combined_lfcc_wavlm = y_test[:min_samples_test_lfcc_wavlm]

# Display the shape of the combined test set
print("X_test_combined_lfcc_wavlm shape:", X_test_combined_lfcc_wavlm.shape)
print("y_test_combined_lfcc_wavlm shape:", y_test_combined_lfcc_wavlm.shape)
