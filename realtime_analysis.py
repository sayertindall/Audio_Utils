# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import librosa as lr

# mp3_dir = './lost.mp3'

# audio, sfreq = lr.load(mp3_dir, sr=44100)
# print(sfreq)
# time = np.arange(0, len(audio)) / sfreq

# fig, ax = plt.subplots()
# ax.plot(time, audio)

# plt.show()


from pydub import AudioSegment

sound = AudioSegment.from_mp3("lost.mp3")

# get raw audio data as a bytestring
raw_data = sound.raw_data
# get the frame rate
sample_rate = sound.frame_rate
# get amount of bytes contained in one sample
sample_size = sound.sample_width
# get channels
channels = sound.channels

# print(len(raw_data))
print(sample_rate, sample_size, channels)