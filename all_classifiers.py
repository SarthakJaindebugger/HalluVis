
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras.layers import Reshape

from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Reshape, Flatten, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from transformer_encoder import TransformerEncoder  # Import your TransformerEncoder implementation

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, SimpleRNN, GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def lstm_model(X_train, X_test, y_train, y_test, epochs, model_save_path='models/lstm_model.keras'):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data to include the batch size
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the LSTM model with reduced L2 regularization strength
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(LSTM(units=64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))  

    # Additional Dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(units=15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=4, activation='softmax'))

    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Train the model with early stopping and learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[lr_scheduler])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Print Classification Report
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, digits=6)
    print(report)

    model.save(model_save_path)
    print(f'Model saved as {model_save_path}')




def cnn_model(X_train, X_test, y_train, y_test, epochs, model_save_path):
    def create_cnn_model(input_shape):
        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(60, activation='relu'))
        return model

    input_shape = (X_train.shape[1], 1)

    # Create a CNN model with the correct input shape
    cnn_model = create_cnn_model(input_shape)

    # Additional Dense layers
    model = Sequential()
    model.add(cnn_model)
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(90, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(56, activation='relu'))

    # Output layer
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Print Classification Report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, digits=6)
    print(report)

    # Save the trained model
    model.save(model_save_path)
    print(f'Model saved as {model_save_path}')


def transformer_model(X_train, X_test, y_train, y_test, epochs, model_save_path):
    def create_transformer_model(input_shape):
        encoder = TransformerEncoder(intermediate_dim=120, num_heads=4)

        input_layer = Input(shape=input_shape)

        # Reshape input data to add channel dimension
        x = Reshape((input_shape[0], 1))(input_layer)

        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)

        x = encoder(x)
        x = Flatten()(x)
        x = Dense(60, activation=None)(x)

        model_transformer = tf.keras.Model(inputs=input_layer, outputs=x)

        return model_transformer

    input_shape = (X_train.shape[1],)  # Adjusted input shape

    # Create a transformer model with the correct input shape
    transformer_model = create_transformer_model(input_shape)

    input_sequence = Input(shape=(X_train.shape[1],))  # Adjusted input shape
    x = transformer_model(input_sequence)

    x = Dense(200, activation='relu', name="dense1")(x)
    x = BatchNormalization()(x)
    x = Dense(90, activation='relu', name="dense2")(x)
    x = Dropout(0.8)(x)
    x = Dense(56, activation='relu', name="dense3")(x)

    output = Dense(4, activation='softmax', name="output")(x)

    model_transformer = tf.keras.Model(inputs=input_sequence, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.5)

    model_transformer.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_transformer.summary()

    # Train the model
    model_transformer.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    # Evaluate the model on the test set
    test_loss, test_accuracy = model_transformer.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    
    # Predict the labels for test data
    y_pred = np.argmax(model_transformer.predict(X_test), axis=-1)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=6))

    # Save the trained model
    model_transformer.save(model_save_path)
    print(f'Model saved as {model_save_path}')



def gru_model(X_train, X_test, y_train, y_test, epochs, model_save_path):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data to include the batch size
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the GRU model
    model = Sequential()
    model.add(GRU(units=64, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(GRU(units=64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))  

    # Additional Dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(units=15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=4, activation='softmax'))

    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Train the model with early stopping and learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[lr_scheduler])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Print Classification Report
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, digits=6)
    print(report)

    model.save(model_save_path)
    print(f'Model saved as {model_save_path}')





def rnn_model(X_train, X_test, y_train, y_test, epochs, model_save_path):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data to include the batch size
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=64, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(SimpleRNN(units=64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))  

    # Additional Dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(units=15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=4, activation='softmax'))

    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Train the model with early stopping and learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[lr_scheduler])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Print Classification Report
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, digits=6)
    print(report)

    model.save(model_save_path)
    print(f'Model saved as {model_save_path}')










from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def svm_model(X_train, X_test, y_train, y_test):
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

    # Initialize the Support Vector Machine classifier
    model = SVC(kernel='linear', random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Fit LabelEncoder on training labels and use it to inverse transform
    le = LabelEncoder()
    le.fit(y_train)
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_test_original, y_pred_original)

    # Visualize the confusion matrix using Seaborn's heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='g', 
                xticklabels=['Violence', 'Non Violence'],
                yticklabels=['Violence', 'Non Violence'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    weighted_f1 = f1_score(y_test_original, y_pred_original, average='weighted')
    print(f'Weighted F1 Score: {weighted_f1}')
    
    # Generate classification report
    class_report = classification_report(y_test_original, y_pred_original, digits=6)
    print('Classification Report:\n', class_report)






from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    # Fit LabelEncoder on training labels and use it to inverse transform
    le = LabelEncoder()
    le.fit(y_train)
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_test_original, y_pred_original)

    # Visualize the confusion matrix using Seaborn's heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='g', 
                xticklabels=['Violence', 'Non Violence'],
                yticklabels=['Violence', 'Non Violence'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')
    
    weighted_f1 = f1_score(y_test_original, y_pred_original, average='weighted')
    print(f'Weighted F1 Score: {weighted_f1}')
    
    # Generate classification report
    class_report = classification_report(y_test_original, y_pred_original, digits=6)
    print('Classification Report:\n', class_report)
