# Wake Up

The aim of this project is to generate sentences in order to use them as WakeUp command.

## Architecture

### Generative Network (GN)

The input will be a random numpy array and the output will be a sequence (of fixed or variable length) of spectrogram's.

[Image of the architecture]

### Discriminant Network (DN)

The input will be a a sequence of spectrogram's and the output will be the probability of being a sentence.

[Image of the architecture]


### Encoder-Decoder (ED)

The input of the encoder will be a sequence of spectrogram's. The output will be one or more different phonemic transcriptions with it's respective probability.

For example we could use: https://github.com/mozilla/DeepSpeech/

[Image of the architecture]


## Procedure
1. Train the DN with words and noise

Loop

2. Evaluate GN with random inputs

3. Train the GN from DN output from previous step input

4. Train the GM with real words and words from the 2nd step

End Loop

At this point GN is able to generate audio that sounds like words

5. Train the ED with audio and transcriptions from certain language. (Already done)

6. Train the GN in order to maximize the probability of the ED. We could add a Language Model here so words not in the dictionary have 0 probability except the brand.

At this point GN is able to generate audio that sounds like words and it's very easy to recognize for the ED
