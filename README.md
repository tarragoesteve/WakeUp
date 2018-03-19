# Wake Up
The aim of this project is to generate Wake Up words.

##Architecture
###Encoder-Decoder
The input of the encoder will be a sequence of spectrogram's. The output will be one or more different phonemic transcriptions with it's respective probability.
[Image of the architecture]

###Generative
The input will be a random numpy array and the output will be a sequence of spectrogram's.

[Image of the architecture]

##How to
1. Train the Encodere-decorder with audio and phonemic transcriptions from certain language. Using cross entropy lost with the correct phonemic transcription.
2. Train the Generative Network in order to maximize the probability of the Encoder-decoder.