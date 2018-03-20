import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


#import urllib

#testfile = urllib.URLopener()
#testfile.retrieve("http://www.thesoundarchive.com/ringtones/Ms_Pacman_Death.wav", "./Ms_Pacman_Death.wav")

rate, data = wavfile.read("/home/esteve/PycharmProjects/WakeUp/Code/Ms_Pacman_Death.wav")
#rate = mostres/sec

f, t, Sxx = signal.spectrogram(x=data, fs=rate)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()