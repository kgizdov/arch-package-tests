#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
print('Loading model test.h5')
model = load_model('test.h5')
print('Predicting on random input')
proba = model.predict_proba(np.random.normal(size = (1,28,28)))
print('Result is:\n{}'.format(proba))
print('Success!')
