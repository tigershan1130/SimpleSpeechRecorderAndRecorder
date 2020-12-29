import os
import pathlib
import sys
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import json
import librosa

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set seed for experiment repdoucibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
SAMPLE_RATE = 16000
EPOCHS = 20


data_dir  = pathlib.Path('mini_speech_commands')


if not (data_dir.exists()):
    print("path does not exist")
    sys.exit(0)

commands = np.array(os.listdir(str(data_dir)))
commands = commands[commands != 'README.md']

print(commands)

findPath = str(data_dir) +'/*/*.wav'

filenames = glob.glob(findPath)
filenames = random.sample(filenames, len(filenames))

num_samples = len(filenames)

train_files = filenames[:6400]
val_files = filenames[6400:6400+800]
test_files = filenames[-800:]
# break our data files size up
print('Number of total examples:', num_samples)
print('Number of examples per label:', len(os.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

def plot_spectrogram(spectrogram, waveform):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform)
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    #plot_spectrogram(spectrogram.numpy(), axes[1])

    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    X = np.arange(16000, step=height + 1)
    Y = range(height)
    axes[1].pcolormesh(X, Y, log_spec)

    axes[1].set_title('Spectrogram')
    plt.show()

def decode_audio(data_path, sr):
    wav, sr = librosa.load(data_path,sr = sr)
    return wav

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path, sr):
    label = get_label(file_path)
    waveform = decode_audio(file_path, sr)

    return waveform, label

def get_spectrogram(wav, sr):
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)

    waveform = tf.cast(wav, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)
    return spectrogram.numpy()

def get_spectrogram_and_label_id(audio, label, sr):
    spectrogram = get_spectrogram(audio, sr)

    # reshape our input from 124/129 to 124,129,1 where 1 represent number of channels.
    spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)

    

    label_id = tf.argmax(label == commands)
    #print('label: ' , label,  commands, label_id)
    return spectrogram, label_id

def preprocess_dataset(files, sr):

    spectros = []
    labels = []

    for data in files:        
        wav, labelID = get_waveform_and_label(data, sr) 
        spectrogram, label_id = get_spectrogram_and_label_id(wav, labelID, sr)
    
        spectros.append(spectrogram)
        labels.append(label_id)

    x_data = np.array(spectros, dtype= 'float32')
    y_data = np.array(labels, dtype= 'float32')

    return x_data, y_data

# prepare training data.
train_spectros, train_labels = preprocess_dataset(train_files, 16000)

val_spectros, val_labels = preprocess_dataset(val_files, 16000)

test_spectros, test_labels = preprocess_dataset(test_files, 16000)

input_shape = train_spectros[0].shape

norm_layer = preprocessing.Normalization()
norm_layer.adapt(train_spectros)

# Construct and compile an instance of customModel
if(os.path.exists('arch.json') and os.path.exists('weights.h5')):

    arch = open('arch.json').read()
    model = tf.keras.models.model_from_json(arch)
    model.load_weights('weights.h5')
    arch = json.loads(arch)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

else:
    model = models.Sequential([
            layers.Input(shape=input_shape),
            preprocessing.Resizing(64, 64), 
            norm_layer,
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(commands)),
    ])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(train_spectros, train_labels, validation_data= (val_spectros, val_labels), epochs=EPOCHS, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),)

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    
    model_json = model.to_json()

    with open('arch.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('weights.h5')

    print("Saved model to disk")

loss, accuracy = model.evaluate(test_spectros, test_labels)

# print(len(test_spectros))

# y_pred = np.argmax(model.predict(test_spectros), axis=1)
# y_true = test_labels

# print(y_true)
# print(y_pred)

# test_acc = (y_pred == y_true).Count() / len(y_true)
# print(f'Test set accuracy: {test_acc:.0%}')


#audio_binary = tf.io.read_file('output.wav')
#waveform = decode_audio(audio_binary)


# spectgram = get_spectrogram(waveform)
# spectgram = tf.expand_dims(spectgram, -1)
# prediction = model.predict(np.array([spectgram]))
# print(prediction)

# predictionIndex = np.argmax(prediction, axis = 1) 

# print(commands[predictionIndex])