#!/usr/bin/env python3

# to be able to run this:
# sudo apt-get install python3 python3-pip
# pip3 install --user matplotlib seaborn numpy h5py tensorflow keras

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc

import matplotlib as mpl
mpl.use('Agg')

# numerical library
import numpy as np
import pandas as pd
import h5py

#import plaidml.keras
#import plaidml
#plaidml.keras.install_backend()

from keras.layers import Input, Dense, Layer, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import binary_crossentropy, mean_squared_error

import keras as K
from utils import LayerNormalization

class Classify(object):
  '''
  Loads a classifier network, trained by ../keras_gan.py and performs inference on input data.
  '''

  def __init__(self, filename = '../network/gan_disc_30000'):
    '''
    Initialise the network.

    :param filename: Name of the network file, without extension.
    '''
    self.filename = filename
    self.disc = None
    self.load(filename)

  '''
  '''
  def read_input_from_files(self, filename = 'input_ee.h5'):
    self.file = pd.HDFStore(filename, 'r')
    self.n_dimensions = self.file['df'].shape[1]-3
    self.col_signal = self.file['df'].columns.get_loc('sample')
    self.col_syst = self.file['df'].columns.get_loc('syst')
    self.col_weight = self.file['df'].columns.get_loc('weight')
    self.sigma = self.file['df'].std(axis = 0).drop(['sample', 'syst', 'weight'], axis = 0)
    self.mean = self.file['df'].mean(axis = 0).drop(['sample', 'syst', 'weight'], axis = 0)
    self.sumWSignal = self.file['df'].loc[self.file['df']['sample'] == 1, 'weight'].sum()
    self.sumWBkg = self.file['df'].loc[self.file['df']['sample'] == 0, 'weight'].sum()
    self.sumW = self.sumWSignal + self.sumWBkg

  def plot_input_correlations(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    nominal = self.file['test_nominal'].iloc[:, 0]
    x = self.file['df'][nominal].drop(['sample', 'syst', 'weight'], axis = 1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(np.corrcoef(x, rowvar = 0),
                cmap="YlGnBu", cbar = True, linewidths=.5, square = True,
                xticklabels = np.arange(0, x.shape[1]), yticklabels = np.arange(0, x.shape[1]),
                annot=True, fmt=".2f")
    ax.set(xlabel = '', ylabel = '', title = 'Correlation between input variables');
    plt.savefig(filename)
    plt.close("all")

  def plot_scatter_input(self, var1, var2, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    nominal = self.file['test_nominal'].iloc[:, 0]
    x = self.file['df'].loc[nominal, (var1, var2)]
    y = self.file['df'].loc[nominal, 'sample']
    g = sns.scatterplot(x = x.loc[:, var1], y = x.loc[:, var2], hue = y,
                        markers = ["^", "v"], legend = "brief", ax = ax)
    #ax.legend(handles = ax.lines[::len(x)+1], labels = ["Background", "Signal"])
    g.axes.get_legend().texts[0] = "Background"
    g.axes.get_legend().texts[1] = "Signal"
    ax.set(xlabel = var1, ylabel = var2, title = 'Scatter plot')
    plt.savefig(filename)
    plt.close("all")

  def plot_discriminator_output(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_signal = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = True): out_signal.extend(self.disc.predict(x))
    out_signal = np.array(out_signal)
    out_bkg = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = False): out_bkg.extend(self.disc.predict(x))
    out_bkg = np.array(out_bkg)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    bins = np.linspace(np.amin(out_signal), np.amax(out_signal), 10)
    sns.distplot(out_signal, bins = bins,
                 kde = False, label = "Test signal", norm_hist = True, hist = True,
                 hist_kws={"histtype": "step", "linewidth": 2, "color": "r"})
    sns.distplot(out_bkg, bins = bins,
                 kde = False, label = "Test bkg.", norm_hist = True, hist = True,
                 hist_kws={"histtype": "step", "linewidth": 2, "color": "b"})
    ax.set(xlabel = 'NN output', ylabel = 'Events', title = '');
    ax.legend(frameon = False)
    plt.savefig(filename)
    plt.close("all")

  def print_output(self):
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_signal = []
    for x,w,y,s in self.get_batch(origin = 'test'): out_signal.extend(self.disc.predict(x))
    out_signal = np.array(out_signal)
    print(out_signal)

  def get_batch(self, origin = 'train', **kwargs):
    filt = np.ones(self.file['df'].shape[0], dtype = 'bool')
    if 'syst' in kwargs and not kwargs['syst']:
      filt = filt & self.file['%s_%s' % (origin, 'nominal')].iloc[:, 0].values
    elif 'syst' in kwargs and kwargs['syst']:
      filt = filt & self.file['%s_%s' % (origin, 'syst')].iloc[:,0].values

    if 'signal' in kwargs and kwargs['signal']:
      filt = filt & self.file['%s_%s' % (origin, 'sig')].iloc[:,0].values
    elif 'signal' in kwargs and not kwargs['signal']:
      filt = filt & self.file['%s_%s' % (origin, 'bkg')].iloc[:,0].values

    filt = np.where(filt)[0]

    rows = np.random.permutation(filt)
    N = len(rows)

    if 'noStop' in kwargs and kwargs['noStop']: # do it forever: whenever the for loop runs out, reset it
      i = 0
      while True:
        r = rows[i*self.n_batch : (i+1)*self.n_batch]
        r = sorted(r)
        df = self.file.select('df', where = 'index = r')
        x_batch = (df.drop(['weight', 'sample', 'syst'], axis = 1) - self.mean)/self.sigma
        y_batch = df.loc[:, 'sample']
        #wmask = ((y_batch == 1)*self.sumWBkg + (y_batch == 0)*self.sumWSig)/self.sumW
        x_batch_w = df.loc[:, 'weight'] #*wmask
        s_batch = df.loc[:, 'syst']
        yield x_batch, x_batch_w, y_batch, s_batch
        i += 1
        if i >= int(N/self.n_batch):
          rows = np.random.permutation(filt)
          i = 0
    else: # do it once over the entire set
      for i in range(0, int(N/self.n_batch)):
        r = rows[i*self.n_batch : (i+1)*self.n_batch]
        r = sorted(r)
        df = self.file.select('df', where = 'index = r')
        x_batch = (df.drop(['weight', 'sample', 'syst'], axis = 1) - self.mean)/self.sigma
        y_batch = df.loc[:, 'sample']
        #wmask = ((y_batch == 1)*self.sumWBkg + (y_batch == 0)*self.sumWSig)/self.sumW
        x_batch_w = df.loc[:, 'weight'] #*wmask
        s_batch = df.loc[:, 'syst']
        yield x_batch, x_batch_w, y_batch, s_batch
      
  '''
  Load stored network
  '''
  def load(self, disc_filename):
    json_file = open('%s.json' % disc_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.disc = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
    self.disc.load_weights("%s.h5" % disc_filename)

    self.disc_input = K.layers.Input(shape = (self.n_dimensions,), name = 'disc_input')

    self.disc.compile(loss = K.losses.binary_crossentropy, optimizer = K.optimizers.Adam(lr = 1e-5), metrics = [])

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Given input data, perform inference on it using a trained discriminator.')
  parser.add_argument('--filename', dest='filename', action='store',
                    default='network/gan_disc_30000',
                    help='Name of the file without extension, where one could find the trained network. (default: "network/gan_disc_30000")')
  parser.add_argument('--input-data', dest='input', action='store',
                    default='input_ee.h5',
                    help='Name of the file from where to read the input. (default: "input_ee.h5")')
  args = parser.parse_args()

  network = Classify(filename = args.filename)

  # read it from disk
  network.read_input_from_files(filename = args.input)

  var = list(network.file['df'].columns.values)
  var.remove('syst')
  var.remove('sample')
  var.remove('weight')

  network.print_output()

if __name__ == '__main__':
  main()

