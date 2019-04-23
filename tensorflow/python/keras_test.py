#!/usr/bin/env python

import numpy as np
from keras.models import load_model
model = load_model('keras_locallyconnected1d.h5')
proba = model.predict_proba(np.random.normal(size = (10,6,1)))
print('Result is:\n{}'.format(proba))
print('Success!')
