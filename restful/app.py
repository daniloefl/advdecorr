#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import connexion

# numerical library
import numpy as np
import tensorflow as tf
import keras as K

'''
Loads a classifier network, trained by ../keras_gan.py and performs inference on input data.
'''
class Network():
  '''
  Initialise the network.
  '''
  def __init__(self, filename = "disc"):
    self.filename = filename
    self.disc = None
    self.load(self.filename)

  '''
  Load stored network
  '''
  def load(self, disc_filename):
    json_file = open('%s.json' % disc_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.disc = K.models.model_from_json(loaded_model_json)
    self.disc.load_weights("%s.h5" % disc_filename)
    self.disc.compile(loss = K.losses.binary_crossentropy, optimizer = K.optimizers.Adam(lr = 1e-5), metrics = [])
    self.graph = tf.get_default_graph()

  '''
  Make a prediction
  '''
  def predict(self, x):
    with self.graph.as_default():
      y = self.disc.predict(x)
    return y

NN = Network()

'''
  Interface with client.
'''
def post_classify(sample):
    out = {}
    x = np.array([[float(sample['A']), float(sample['B'])]])
    out['i'] = sample['i']
    out['A'] = sample['A']
    out['B'] = sample['B']
    out['pvalue'] = str(float(NN.predict(x)))
    return out

app = connexion.App(__name__, specification_dir = './')
app.add_api('api_configuration.yaml')
application = app.app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

