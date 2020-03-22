import matplotlib.pyplot as plt
import numpy as np
import wave
from sys import argv

import numpy as np
import contextlib


def pcm2float(sig, dtype='float64'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

with wave.open(argv[1]) as w:
    framerate = w.getframerate()
    frames = w.getnframes()
    channels = w.getnchannels()
    width = w.getsampwidth()
    print('sampling rate:', framerate, 'Hz')
    print('length:', frames, 'samples')
    print('channels:', channels)
    print('sample width:', width, 'bytes')
    
    data = w.readframes(frames)
	
sig = np.frombuffer(data, dtype='<i2').reshape(-1, channels)
normalized = pcm2float(sig, 'float32')

for val in normalized:
	print(val)
