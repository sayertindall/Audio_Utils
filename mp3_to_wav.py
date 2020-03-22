from os import path
from pydub import AudioSegment

# files                                                                         
src = "fileName.mp3"
dst = "fileOutputName.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")