import pyaudio
import wave
import librosa
import tkinter as tk
import threading
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 500
CHUNK_SIZE = 1024
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 3
TRIM_TO_SECONDS = 1

class RecordingApp():

    frames = []

    def __init__(self, master):
        self.isRecording = False
        self.button1 = tk.Button(main, text = 'recording', command = self.startrecording)
        self.button2 = tk.Button(main, text = 'stop', command = self.stoprecording)

        self.button1.pack()
        self.button2.pack()


    def startrecording(self):
        self.p = pyaudio.PyAudio()  
        self.stream = self.p.open(format=FORMAT,channels=CHANNELS,rate=RATE,frames_per_buffer=CHUNK_SIZE,input=True)
        self.isrecording = True
        
        print('Recording')
        t = threading.Thread(target=self.record)
        t.start()

    def trimAudio(self, path, trimSecs):

        
        sr, waveData = wavfile.read(path)
        duration = len(waveData) / sr
        time = np.arange(0, duration, 1/sr)

        if(duration < trimSecs):
        
            return            
        else:
            print(waveData)
            print(np.max(waveData))
            print(np.min(waveData))

            offset = trimSecs / 2 * sr 

            middleValue = np.max(waveData)

            if(abs(np.min(waveData)) > middleValue ):
                middleValue = np.min(waveData)

            index = np.where(waveData == middleValue)

            print(index)

            end = index[0] + int(offset)
            start = index[0] - int(offset)
            if(start < 0):
                start = 0

            if(end >= len(waveData)):
                end = len(waveData)-1


            print(start[0])
            print(end[0])

            trimedData = waveData[start[0]: end[0]]


            print(trimedData)

            print(len(trimedData))
            #time = np.arange(len(trimedData) / sr)

            
           # plt.plot(time, trimedData)
            #plt.xlabel('Time [s]')
            #plt.ylabel('amplitude')
            #plt.title(self.filename)
           # plt.show()    

            wavfile.write(self.filename, sr, trimedData)

    def stoprecording(self):
        self.isrecording = False
        print('recording complete')
        #print(self.frames)
        self.filename=input('the filename?')
        self.filename = self.filename+".wav"
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        main.destroy()

        self.trimAudio(self.filename, 1)

    def record(self):
       
        while self.isrecording:
            data = self.stream.read(CHUNK_SIZE)
            self.frames.append(data)

main = tk.Tk()
main.title('recorder')
main.geometry('200x50')
app = RecordingApp(main)

main.mainloop()