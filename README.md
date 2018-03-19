# Wake Up

The aim of this project is to generate Wake Up words.

##Architecture

###Generative Network (GN)

The input will be a random numpy array and the output will be a sequence of spectrogram's.

[Image of the architecture]

###Discriminant Network (DN)

Tells if a sound is a word or not


###Encoder-Decoder (ED)

The input of the encoder will be a sequence of spectrogram's. The output will be one or more different phonemic transcriptions with it's respective probability.
[Image of the architecture]


##How to
1. Train the DN with words and noise

Loop

2. Evaluate GN with random inputs

3. Train the GN from DN output from previous step input

4. Train the GM with real words and words from the 2nd step

End Loop

At this point GN is able to generate audio that sounds like words

5. Train the ED with audio and phonemic transcriptions from certain language. Using cross entropy lost with the correct phonemic transcription.

6. Train the GN in order to maximize the probability of the ED.

At this point GN is able to generate audio that sounds like words and it's very easy to recognize for the ED