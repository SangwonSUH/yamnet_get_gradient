import params as params
import yamnet as yamnet_model
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')

import soundfile as sf
import numpy as np
wav_data, sr = sf.read('Y_CZzm6jUxbo_30.000_40.000.wav', dtype=np.int16)
assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

import resampy
# Convert to mono and the sample rate expected by YAMNet.
if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
if sr != params.SAMPLE_RATE:
    waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

# Predict YAMNet classes.
# Second output is log-mel-spectrogram array (used for visualizations).
# (steps=1 is a work around for Keras batching limitations.)
scores, patches, mel_spec = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)

prediction = np.mean(scores, axis=0)

import keras.backend as K
import keras
import tensorflow as tf
y_c = yamnet.output[0][0, prediction.argmax()]
conv_output = yamnet.get_layer('layer14/pointwise_conv').output
grads = K.gradients(y_c, conv_output)[0]
gradient_function = K.function([yamnet.input], [conv_output, grads])

session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

output, grads_val = gradient_function([np.reshape(waveform, [1, -1])])
output.shape
grads_val.shape