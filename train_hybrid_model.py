import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Bidirectional, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import logging
import random
from scipy.signal import butter, lfilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = "data"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function for data augmentation
def augment_audio(y, sr, augmentation_type=None):
    """Apply various augmentation techniques to audio data."""
    if augmentation_type is None:
        augmentation_type = random.choice(['pitch', 'speed', 'noise', 'shift', 'none'])
    
    if augmentation_type == 'pitch':
        # Pitch shift (up or down by 0-2 semitones)
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    elif augmentation_type == 'speed':
        # Time stretching (speed up or slow down by 0.8-1.2 factor)
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y, rate=rate)
    
    elif augmentation_type == 'noise':
        # Add random noise
        noise_factor = random.uniform(0.005, 0.015)
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    elif augmentation_type == 'shift':
        # Time shift (shift by -0.5 to 0.5 seconds)
        shift = int(random.uniform(-0.5, 0.5) * sr)
        if shift > 0:
            return np.pad(y, (shift, 0), mode='constant')[0:len(y)]
        else:
            return np.pad(y, (0, -shift), mode='constant')[0:len(y)]
    
    else:  # 'none'
        return y

# Function to apply bandpass filter
def bandpass_filter(data, lowcut=300, highcut=8000, fs=22050, order=5):
    """Apply bandpass filter to focus on speech frequencies."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Function to extract features from audio files
def extract_features(file_path, augment=False):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Apply bandpass filter to focus on speech frequencies
        y = bandpass_filter(y)
        
        # Apply data augmentation if requested
        if augment:
            y = augment_audio(y, sr)
        
        # If audio is too short, pad it
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
        
        # If audio is too long, truncate it
        if len(y) > sr * 5:  # Limit to 5 seconds
            y = y[:sr * 5]
        
        # Extract mel spectrogram for Hybrid CNN-LSTM
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to expected input shape
        if mel_spec_db.shape[1] < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])), 'constant')
        else:
            mel_spec_db = mel_spec_db[:, :128]
        
        return mel_spec_db
    
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

# Function to load dataset
def load_dataset(augment_data=True):
    features = []
    labels = []
    
    # Process each dataset
    datasets = ['ravdess', 'crema', 'tess', 'savee']
    total_files = 0
    processed_files = 0
    
    # First count total files
    for dataset in datasets:
        dataset_path = os.path.join(DATA_PATH, dataset)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} does not exist. Skipping.")
            continue
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    total_files += 1
    
    logger.info(f"Found {total_files} audio files across all datasets")
    
    # Now process each dataset
    for dataset in datasets:
        dataset_path = os.path.join(DATA_PATH, dataset)
        
        if not os.path.exists(dataset_path):
            continue
        
        logger.info(f"Processing dataset: {dataset}")
        dataset_files = 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    processed_files += 1
                    dataset_files += 1
                    
                    if processed_files % 100 == 0:
                        logger.info(f"Processed {processed_files}/{total_files} files ({(processed_files/total_files)*100:.1f}%)")
                    
                    # Extract emotion from filename or directory structure
                    emotion = None
                    
                    # RAVDESS format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                    if dataset == 'ravdess':
                        emotion_code = int(file.split('-')[2])
                        emotion_map = {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 
                                      5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
                        emotion = emotion_map.get(emotion_code)
                    
                    # CREMA-D format: ActorID_Sentence_Emotion_Intensity.wav
                    elif dataset == 'crema':
                        emotion_code = file.split('_')[2]
                        emotion_map = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 
                                      'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
                        emotion = emotion_map.get(emotion_code)
                    
                    # TESS format: OAF_emotion_word.wav or YAF_emotion_word.wav
                    elif dataset == 'tess':
                        if 'OAF_' in file:
                            parts = file.split('_')
                            if len(parts) >= 2:
                                emotion = parts[1].lower()
                        elif 'YAF_' in file:
                            parts = file.split('_')
                            if len(parts) >= 2:
                                emotion = parts[1].lower()
                        else:
                            # Extract from directory name
                            dir_name = os.path.basename(root).lower()
                            if 'angry' in dir_name:
                                emotion = 'angry'
                            elif 'disgust' in dir_name:
                                emotion = 'disgust'
                            elif 'fear' in dir_name:
                                emotion = 'fear'
                            elif 'happy' in dir_name:
                                emotion = 'happy'
                            elif 'neutral' in dir_name:
                                emotion = 'neutral'
                            elif 'sad' in dir_name:
                                emotion = 'sad'
                            elif 'surprise' in dir_name or 'pleasant_surprised' in dir_name or 'pleasant_surprise' in dir_name:
                                emotion = 'surprise'
                    
                    # SAVEE format: emotion_statement_repetition.wav
                    elif dataset == 'savee':
                        # First check for two-letter emotion codes
                        if file.startswith('sa') and len(file) > 2 and file[2] in ['0', '1', '_']:
                            emotion = 'sad'
                        elif file.startswith('su') and len(file) > 2 and file[2] in ['0', '1', '_']:
                            emotion = 'surprise'
                        else:
                            # Then check for single-letter emotion codes
                            emotion_code = file[0]
                            emotion_map = {'n': 'neutral', 'h': 'happy', 
                                          'a': 'angry', 'f': 'fear', 'd': 'disgust'}
                            emotion = emotion_map.get(emotion_code)
                    
                    # Skip if emotion not identified
                    if not emotion or emotion not in EMOTIONS:
                        logger.warning(f"Could not identify emotion for {file_path}. Skipping.")
                        continue
                    
                    # Extract features
                    mel_features = extract_features(file_path)
                    
                    if mel_features is not None:
                        features.append(mel_features)
                        labels.append(emotion)
                        
                        # Add augmented versions if requested
                        if augment_data:
                            # Add 1-2 augmented versions for underrepresented emotions
                            if emotion in ['disgust', 'fear', 'surprise']:
                                num_augmentations = 2
                            else:
                                num_augmentations = 1
                                
                            for _ in range(num_augmentations):
                                mel_aug = extract_features(file_path, augment=True)
                                
                                if mel_aug is not None:
                                    features.append(mel_aug)
                                    labels.append(emotion)
        
        logger.info(f"Finished processing {dataset} dataset: {dataset_files} files")
    
    logger.info(f"Total processed files: {len(features)}")
    # Count emotions
    emotion_counts = {}
    for emotion in labels:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    for emotion, count in emotion_counts.items():
        logger.info(f"Emotion '{emotion}': {count} samples")
    
    return np.array(features), np.array(labels)

# Function to build improved Hybrid CNN-LSTM model
def build_hybrid_model(input_shape=(128, 128, 1), num_classes=7):
    # CNN part
    cnn_input = Input(shape=input_shape)
    
    # First convolutional block with residual connection
    x = Conv2D(32, (3, 3), padding='same')(cnn_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Second convolutional block with residual connection
    x_res = Conv2D(64, (1, 1), strides=(2, 2))(cnn_input)  # Shortcut connection
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])  # Add residual connection
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Reshape for LSTM
    x = tf.keras.layers.Reshape((-1, 64 * (input_shape[0] // 4)))(x)
    
    # LSTM part
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    
    # Dense layers
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=cnn_input, outputs=outputs)
    
    # Compile model with a lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train Hybrid CNN-LSTM model
def train_hybrid_model():
    # Load dataset
    logger.info("Loading dataset with augmentation...")
    features, labels = load_dataset(augment_data=True)
    
    if len(features) == 0:
        logger.error("No features extracted. Exiting.")
        return
    
    logger.info(f"Dataset loaded with {len(features)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Reshape features for CNN input (samples, height, width, channels)
    X_mel_cnn = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_mel_cnn, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        ModelCheckpoint(filepath=os.path.join(SAVE_DIR, 'hybrid_{epoch:02d}-{val_accuracy:.4f}.h5'), 
                        monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train Hybrid CNN-LSTM model
    logger.info("Training Hybrid CNN-LSTM model...")
    hybrid_model = build_hybrid_model(input_shape=(128, 128, 1), num_classes=len(EMOTIONS))
    logger.info(f"Hybrid model summary: {hybrid_model.summary()}")
    hybrid_history = hybrid_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    hybrid_model.save(os.path.join(SAVE_DIR, 'hybrid_model.h5'))
    logger.info("Hybrid model training completed and saved")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hybrid_history.history['accuracy'], label='Training')
    plt.plot(hybrid_history.history['val_accuracy'], label='Validation')
    plt.title('Hybrid CNN-LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(hybrid_history.history['loss'], label='Training')
    plt.plot(hybrid_history.history['val_loss'], label='Validation')
    plt.title('Hybrid CNN-LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'hybrid_training_history.png'))
    
    logger.info("Hybrid CNN-LSTM model training completed. Model saved to 'models/hybrid_model.h5'")

if __name__ == "__main__":
    train_hybrid_model() 