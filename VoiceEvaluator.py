import pyaudio
import wave
import tensorflow as tf
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 500
CHUNK_SIZE = 1024
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 3
TRIM_TO_SECONDS = 1

WAVE_OUTPUT_FILENAME = 'output.wav'

commands = ['down','go','left','no','right','stop','up','yes']

def loadKerasModel():
    arch = open('arch.json').read()
    model = tf.keras.models.model_from_json(arch)
    model.load_weights('weights.h5')
    arch = json.loads(arch)

    return model

# load wav
def loadWav(data_path):
    
    wav, sr = librosa.load(data_path,sr = 16000)

    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)

    waveform = tf.cast(wav, tf.float32)

    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)
    
    spectrogram = spectrogram.numpy()

    # reshape our input from 124/129 to 124,129,1 where 1 represent number of channels.
    spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)    

    return spectrogram




def predictModel(model, spectrogram):

    listOfData = []
    listOfData.append(spectrogram)

    predictData = np.array(listOfData, dtype= 'float32')

    result = model.predict(predictData)

    lab = tf.argmax(result, 1)
    return lab



if __name__ == '__main__':
    #load keras model
    model = loadKerasModel()
    spectrogram = loadWav('Test1.wav')
    prediction = model.predict(np.array([spectrogram]))
    predictionIndex = np.argmax(prediction, axis = 1)

    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions is: "{commands[predictionIndex[0]]}"')
    plt.show()
