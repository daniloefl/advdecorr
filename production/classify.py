#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse

app = Flask(__name__)
api = Api(app)

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
  def __init__(self, filename = "ganna_discriminator_30000"):
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

'''
  Class that implements interface with client.
'''
class Classify(Resource):

  '''
  PUT method handling.
  '''
  def put(self):

    parser = reqparse.RequestParser()
    parser.add_argument('i', type=int, help='Item index')
    parser.add_argument('A', type=float, help='Value of A')
    parser.add_argument('B', type=float, help='Value of B')
    samples = parser.parse_args()

    out = {}
    x = np.array([[samples['A'], samples['B']]])
    out[samples['i']] = float(NN.predict(x))
    return out

NN = Network()
api.add_resource(Classify, '/classify')

if __name__ == '__main__':
    app.run(port=5001)

