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

def smoothen(y):
  return y

class GAN(object):
  '''
  Implementation of GAN algorithm to decorrelate a discriminator in a variable S.
  Ref.: https://arxiv.org/pdf/1704.00028.pdf
  Ref.: https://arxiv.org/abs/1701.07875
  Ref.: https://arxiv.org/abs/1406.2661
  The objective of the discriminator is to separate signal (Y = 1) from backgrond (Y = 0). The adv punishes the discriminator
  if it can guess whether the sample is the nominal (N = 1) or a systematic uncertainty (N = 0).

  The discriminator outputs o = D(x), trying to minimize the cross-entropy for the signal/background classification:
  L_{disc. only} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ]

  The adv. estimates a function C(o), where o = D(x), such that the following discriminates between nominal and variation:
  L_{adv.} = E_{nominal}[ - log C(D(x))] + E_{uncertainty}[ - log(1 - C(D(x)))]

  To impose those restrictions and punish the discriminator for leaning about nominal or systematic uncertainties (to decorrelate it),
  the following procedure is implemented:
  1) Pre-train the discriminator n_pretrain batches, so that it can separate signal from background using this loss function:
  L_{disc. only} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ]

  2) Train the adv., fixing the discriminator, in n_adv batches,
     between nominal and uncertainty in the output of the discriminator.
     Both signal and background samples are used here.
     epsilon = batch_size samples of a uniform distribution between 0 and 1
  L_{adv.} = -\lambda_{decorr} { E_{nominal} [ log C(D(x)) ] + E_{uncertainty} [ log(1 - C(D(x))) ] }

  3) Train the discriminator, fixing the adv., in one batch to minimize simultaneously the discriminator cross-entropy and the adversary's.
  L_{all} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ] + \lambda_{decorr} { E_{nominal} [ log C(D(x)) ] + E_{uncertainty} [ log (1 - C(D(x))) ] }

  4) Go back to 2 and repeat this n_iteration times.
  '''

  def __init__(self, n_iteration = 10001, n_pretrain = 0, n_adv = 100,
               n_batch = 128,
               lambda_decorr = 50.0,
               n_eval = 50,
               no_adv = False):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_pretrain: Number of batches to run over to pre-train the discriminator.
    :param n_adv: Number of batches to train the adv. on per batch of the discriminator.
    :param n_batch: Number of samples in a batch.
    :param lambda_decorr: Lambda parameter used to weigh the decorrelation term of the discriminator loss function.
    :param n_eval: Number of batches to train before evaluating metrics.
    :param no_adv: Do not train the adv., so that we can check the impact of the training in an independent discriminator.
    '''
    self.n_iteration = n_iteration
    self.n_pretrain = n_pretrain
    self.n_adv = n_adv
    self.n_batch = n_batch
    self.lambda_decorr = lambda_decorr
    self.n_eval = n_eval
    self.no_adv = no_adv
    self.adv = None
    self.disc = None

  '''
    Create adv. network.
  '''
  def create_adv(self):
    self.advsig_input = Input(shape = (1,), name = 'advsig_input')
    self.advaux_input = Input(shape = (1,), name = 'advaux_input')
    self.adv_input = Input(shape = (1,), name = 'adv_input')
    xc = K.layers.Concatenate()([self.advsig_input, self.adv_input])
    #xc = self.adv_input
    xc = Dense(20, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = Dense(20, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = Dense(1, activation = 'sigmoid')(xc)
    self.adv = Model([self.advsig_input, self.adv_input], xc, name = "adv")
    #self.adv = Model(self.adv_input, xc, name = "adv")
    self.adv.trainable = True
    self.adv.compile(loss = K.losses.binary_crossentropy,
                        optimizer = Adam(lr = 1e-3), metrics = [])

  '''
  Create discriminator network.
  '''
  def create_disc(self):
    self.disc_input = Input(shape = (self.n_dimensions,), name = 'disc_input')

    xd = self.disc_input
    xd = Dense(20, activation = None)(xd)
    xd = K.layers.Activation('tanh')(xd)
    xd = Dense(20, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = Dense(1, activation = 'sigmoid')(xd)
    self.disc = Model(self.disc_input, xd, name = "disc")
    self.disc.trainable = True
    self.disc.compile(loss = K.losses.binary_crossentropy, optimizer = Adam(lr = 1e-3), metrics = [])

  '''
  Create all networks.
  '''
  def create_networks(self):
    if not self.adv:
      self.create_adv()
    if not self.disc:
      self.create_disc()

    self.disc.trainable = False
    self.adv.trainable = True

    self.any_input = Input(shape = (self.n_dimensions,), name = 'any_input')
    self.nominal_input = Input(shape = (self.n_dimensions,), name = 'nominal_input')
    self.syst_input = Input(shape = (self.n_dimensions,), name = 'syst_input')
    self.sig_input = Input(shape = (1,), name = 'sig_input')
    self.aux_input = Input(shape = (1,), name = 'aux_input')

    self.disc.trainable = False
    self.adv.trainable = True
    self.disc_fixed_adv = Model([self.any_input, self.sig_input],
                                [self.adv([self.sig_input, self.disc(self.any_input)])],
                                #[self.adv([self.sig_input, self.disc(self.any_input)])],
                                name = "disc_fixed_adv")
    self.disc_fixed_adv.compile(loss = [K.losses.binary_crossentropy],
                                loss_weights = [self.lambda_decorr],
                                #optimizer = RMSprop(lr = 1e-4), metrics = [])
                                optimizer = Adam(lr = 1e-4, beta_1 = 0, beta_2 = 0.9), metrics = [])

    self.disc.trainable = True
    self.adv.trainable = False
    self.disc_adv_fixed = Model([self.nominal_input, self.any_input, self.sig_input],
                                [self.disc(self.nominal_input),
                                 self.adv([self.sig_input, self.disc(self.any_input)])],
                                 #self.adv([self.sig_input, self.disc(self.any_input)])],
                                name = "disc_adv_fixed")
    self.disc_adv_fixed.compile(loss = [K.losses.binary_crossentropy,
                                        K.losses.binary_crossentropy],
                                   loss_weights = [1.0, -self.lambda_decorr],
                                   #optimizer = RMSprop(lr = 1e-4), metrics = [])
                                   optimizer = Adam(lr = 1e-4, beta_1 = 0, beta_2 = 0.9), metrics = [])


    print("Signal/background discriminator:")
    self.disc.trainable = True
    self.disc.summary()
    print("Adv.:")
    self.adv.trainable = True
    self.adv.summary()
    print("Disc. against adv.:")
    self.disc.trainable = True
    self.adv.trainable = False
    self.disc_adv_fixed.summary()
    print("Adv. against disc.:")
    self.disc.trainable = False
    self.adv.trainable = True
    self.disc_fixed_adv.summary()


  '''
  '''
  def read_input_from_files(self, filename = 'input_preprocessed.h5'):
    self.file = pd.HDFStore(filename, 'r')
    self.n_dimensions = self.file['df'].shape[1]-4
    self.col_signal = self.file['df'].columns.get_loc('sample')
    self.col_syst = self.file['df'].columns.get_loc('syst')
    self.col_weight = self.file['df'].columns.get_loc('weight')
    self.sigma = self.file['df'].std(axis = 0).drop(['sample', 'syst', 'weight', 'train'], axis = 0)
    self.mean = self.file['df'].mean(axis = 0).drop(['sample', 'syst', 'weight', 'train'], axis = 0)
    self.sumWSignal = self.file['df'].loc[self.file['df']['sample'] == 1, 'weight'].sum()
    self.sumWBkg = self.file['df'].loc[self.file['df']['sample'] == 0, 'weight'].sum()
    self.sumW = self.sumWSignal + self.sumWBkg

  def plot_input_correlations(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    nominal = self.file['test_nominal'].iloc[:, 0]
    x = self.file['df'][nominal].drop(['sample', 'syst', 'weight', 'train'], axis = 1)
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

  def plot_discriminator_output_syst(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_signal = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = True, syst = False): out_signal.extend(self.disc.predict(x))
    out_signal = np.array(out_signal)
    out_bkg = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = False, syst = False): out_bkg.extend(self.disc.predict(x))
    out_bkg = np.array(out_bkg)

    out_signal_s = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = True, syst = True): out_signal_s.extend(self.disc.predict(x))
    out_signal_s = np.array(out_signal_s)
    out_bkg_s = []
    for x,w,y,s in self.get_batch(origin = 'test', signal = False, syst = True): out_bkg_s.extend(self.disc.predict(x))
    out_bkg_s = np.array(out_bkg_s)

    Nbins = 10
    bins = np.linspace(0, 1.0, Nbins+1)
    h_signal, be = np.histogram(out_signal, bins = bins)
    e_signal, _ = np.histogram(out_signal**2, bins = bins)
    h_bkg, _ = np.histogram(out_bkg, bins = bins)
    e_bkg, _ = np.histogram(out_bkg**2, bins = bins)

    h_signal_s, _ = np.histogram(out_signal_s, bins = bins)
    e_signal_s, _ = np.histogram(out_signal_s**2, bins = bins)
    h_bkg_s, _ = np.histogram(out_bkg_s, bins = bins)
    e_bkg_s, _ = np.histogram(out_bkg_s**2, bins = bins)

    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, gridspec_kw = {'height_ratios':[4, 1]})
    bel = be[:-1]
    bc = 0.5*be[0:-1] + 0.5*be[1:]
    xbl = bc - be[0:-1]
    xbh = be[1:] - bc
    N = len(e_signal)

    bel = np.concatenate( (np.array([bel[0]]), bel, np.array([be[-1]])) )
    h_signal = np.concatenate( (np.array([0]), h_signal, np.array([0])) )
    h_bkg = np.concatenate( (np.array([0]), h_bkg, np.array([0])) )
    h_signal_s = np.concatenate( (np.array([0]), h_signal_s, np.array([0])) )
    h_bkg_s = np.concatenate( (np.array([0]), h_bkg_s, np.array([0])) )
    e_signal = np.concatenate( (np.array([0]), e_signal, np.array([0])) )
    e_bkg = np.concatenate( (np.array([0]), e_bkg, np.array([0])) )
    e_signal_s = np.concatenate( (np.array([0]), e_signal_s, np.array([0])) )
    e_bkg_s = np.concatenate( (np.array([0]), e_bkg_s, np.array([0])) )
    N += 2

    ax[0].plot(bel, h_signal, color = 'r', linewidth = 2, label = 'Test signal (nominal)', drawstyle = 'steps-post')
    #ax[0].errorbar(bel, h_signal, yerr = e_signal, color = 'r', drawstyle = 'steps-post')
    ax[0].plot(bel, h_bkg, color = 'b', linewidth = 2, label = 'Test bkg. (nominal)', drawstyle = 'steps-post')
    #ax[0].errorbar(bel, h_bkg, yerr = e_bkg, color = 'b', drawstyle = 'steps-post')

    ax[0].plot(bel, h_signal_s, color = 'r', linewidth = 2, linestyle = '--', label = 'Test signal (syst.)', drawstyle = 'steps-post')
    #ax[0].errorbar(bel, h_signal_s, yerr = e_signal_s, color = 'r', drawstyle = 'steps-post')
    ax[0].plot(bel, h_bkg_s, color = 'b', linewidth = 2, linestyle = '--', label = 'Test bkg. (syst.)', drawstyle = 'steps-post')
    #ax[0].errorbar(bel, h_bkg_s, yerr = e_bkg_s, color = 'b', drawstyle = 'steps-post')

    hr_signal = np.divide(h_signal_s, h_signal, out = np.zeros(N), where = h_signal != 0)
    er_signal = e_signal_s*(np.divide(np.ones(N), h_signal, out = np.zeros(N), where = h_signal != 0))**2 + e_signal*(np.divide(h_signal_s, h_signal, where = h_signal != 0)**2)**2
    hr_bkg = np.divide(h_bkg_s, h_bkg, out = np.zeros(N), where = h_bkg != 0)
    er_bkg = e_bkg_s*(np.divide(np.ones(N), h_bkg, out = np.zeros(N), where = h_bkg != 0))**2 + e_bkg*(np.divide(h_bkg_s, h_bkg, out = np.zeros(N), where = h_bkg != 0)**2)**2

    ax[1].plot(bel, hr_signal, color = 'r', linewidth = 2, drawstyle = 'steps-post')
    #ax[1].errorbar(bel, hr_signal, yerr = er_signal, color = 'r', drawstyle = 'steps-post')
    ax[1].plot(bel, hr_bkg, color = 'b', linewidth = 2, drawstyle = 'steps-post')
    #ax[1].errorbar(bel, hr_bkg, yerr = er_bkg, color = 'b', drawstyle = 'steps-post')

    ax[1].set_ylim([0.5, 1.5])
    m = np.amax(np.concatenate( (h_signal, h_bkg, h_signal_s, h_bkg_s) ) )
    ax[0].set_ylim([0.0, 1.1*m])

    ax[0].set_xlim([bins[0], bins[-1]])
    ax[1].set_xlim([bins[0], bins[-1]])

    #plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    ax[1].set(xlabel = 'NN output', ylabel = 'Syst./Nominal', title = '');
    ax[0].set(xlabel = '', ylabel = 'Events', title = '');

    p = ax[1].yaxis.get_label().get_position()
    ax[1].yaxis.get_label().set_position([0.10, p[1]])
    p = ax[0].yaxis.get_label().get_position()
    ax[0].yaxis.get_label().set_position([0.10, p[1]])

    ax[0].legend(frameon = False)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.draw()
    tl = [i.get_text() for i in ax[0].get_yticklabels()]
    tl[0] = ""
    ax[0].set_yticklabels(tl)
    plt.savefig(filename)
    plt.close("all")

  def plot_adv_output(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # use get_continuour_batch to read directly from the file
    out_syst_nominal = []
    for x,w,y,s in self.get_batch(origin = 'test', syst = False): out_syst_nominal.extend(self.adv.predict([y, self.disc.predict(x)]))
    #for x,w,y,s in self.get_batch(origin = 'test', syst = False): out_syst_nominal.extend(self.adv.predict(self.disc.predict(x)))
    out_syst_nominal = np.array(out_syst_nominal)
    out_syst_var = []
    for x,w,y,s in self.get_batch(origin = 'test', syst = True): out_syst_var.extend(self.adv.predict([y, self.disc.predict(x)]))
    #for x,w,y,s in self.get_batch(origin = 'test', syst = True): out_syst_var.extend(self.adv.predict(self.disc.predict(x)))
    out_syst_var = np.array(out_syst_var)
    bins = np.linspace(np.amin(out_syst_nominal), np.amax(out_syst_nominal), 10)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.distplot(out_syst_nominal, bins = bins,
                 kde = False, label = "Test nominal", norm_hist = True, hist = True,
                 hist_kws={"histtype": "step", "linewidth": 2, "color": "r"})
    sns.distplot(out_syst_var, bins = bins,
                 kde = False, label = "Test syst. var.", norm_hist = True, hist = True,
                 hist_kws={"histtype": "step", "linewidth": 2, "color": "b"})
    ax.set(xlabel = 'Critic NN output', ylabel = 'Events', title = '');
    ax.legend(frameon = False)
    plt.savefig(filename)
    plt.close("all")

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
        x_batch = (df.drop(['weight', 'sample', 'syst', 'train'], axis = 1) - self.mean)/self.sigma
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
        x_batch = (df.drop(['weight', 'sample', 'syst', 'train'], axis = 1) - self.mean)/self.sigma
        y_batch = df.loc[:, 'sample']
        #wmask = ((y_batch == 1)*self.sumWBkg + (y_batch == 0)*self.sumWSig)/self.sumW
        x_batch_w = df.loc[:, 'weight'] #*wmask
        s_batch = df.loc[:, 'syst']
        yield x_batch, x_batch_w, y_batch, s_batch
      

  def train(self, prefix, result_dir, network_dir):
    # algorithm:
    # 0) pretrain discriminator
    # 1) Train adv. to guess syst. (freezing discriminator)
    # 2) Train disc. to fool adv. (freezing adv.)
    self.adv_gp_loss_train = np.array([])
    self.adv_loss_train = np.array([])
    self.adv_loss_nom_train = np.array([])
    self.adv_loss_sys_train = np.array([])
    self.disc_loss_train = np.array([])

    iter_any_bkg = self.get_batch(origin = 'train', noStop = True, signal = False)
    iter_any_sig = self.get_batch(origin = 'train', noStop = True, signal = True)
    iter_nom = self.get_batch(origin = 'train', syst = False, noStop = True)
    iter_any = self.get_batch(origin = 'train', noStop = True)
    iter_test_nom = self.get_batch(origin = 'test', syst = False, noStop = True)
    iter_test_any = self.get_batch(origin = 'test', noStop = True)
    iter_test_sys = self.get_batch(origin = 'test', syst = True, noStop = True)

    for epoch in range(self.n_iteration):
      if self.no_adv:
        #print("Training with no adv.", self.no_adv, not self.no_adv, self.lambda_decorr)
        #x_batch_nom, x_batch_nom_w, y_batch_nom, s_batch_nom = next(iter_nom)
        x_batch_any, x_batch_any_w, y_batch_any, s_batch_any = next(iter_any)
        self.disc.trainable = True
        self.disc.train_on_batch([x_batch_any],
                                 [y_batch_any],
                                 sample_weight = [x_batch_any_w])

        n_adv = self.n_adv
        for k in range(0, n_adv):
          self.disc.trainable = False
          self.adv.trainable = True
          x_batch_sig, x_batch_sig_w, y_batch_sig, s_batch_sig = next(iter_any_sig)
          x_batch_bkg, x_batch_bkg_w, y_batch_bkg, s_batch_bkg = next(iter_any_bkg)
          x_batch_any   = np.concatenate( (x_batch_sig, x_batch_bkg), axis = 0)
          x_batch_any_w = np.concatenate( (x_batch_sig_w, x_batch_bkg_w), axis = 0)
          y_batch_any   = np.concatenate( (y_batch_sig, y_batch_bkg), axis = 0)
          s_batch_any   = np.concatenate( (s_batch_sig, s_batch_bkg), axis = 0)
          self.disc_fixed_adv.train_on_batch([x_batch_any, y_batch_any],
                                             [s_batch_any],
                                             sample_weight = [x_batch_any_w])

      if not self.no_adv:
        #print("Training with adv.", self.no_adv, not self.no_adv, self.lambda_decorr)
        # step 0 - pretraining
        if epoch < self.n_pretrain:
          x_batch_any, x_batch_any_w, y_batch_any, s_batch_any = next(iter_any)
          self.disc.trainable = True
          self.disc.train_on_batch([x_batch_any],
                                   [y_batch_any],
                                   sample_weight = [x_batch_any_w])
          n_adv = self.n_adv
          for k in range(0, n_adv):
            self.disc.trainable = False
            self.adv.trainable = True
            x_batch_sig, x_batch_sig_w, y_batch_sig, s_batch_sig = next(iter_any_sig)
            x_batch_bkg, x_batch_bkg_w, y_batch_bkg, s_batch_bkg = next(iter_any_bkg)
            x_batch_any   = np.concatenate( (x_batch_sig, x_batch_bkg), axis = 0)
            x_batch_any_w = np.concatenate( (x_batch_sig_w, x_batch_bkg_w), axis = 0)
            y_batch_any   = np.concatenate( (y_batch_sig, y_batch_bkg), axis = 0)
            s_batch_any   = np.concatenate( (s_batch_sig, s_batch_bkg), axis = 0)
            self.disc_fixed_adv.train_on_batch([x_batch_any, y_batch_any],
                                               [s_batch_any],
                                               sample_weight = [x_batch_any_w])

        if epoch >= self.n_pretrain:
          # step adv.
          self.disc.trainable = False
          self.adv.trainable = True
          n_adv = self.n_adv
          for k in range(0, n_adv):
            self.disc.trainable = False
            self.adv.trainable = True
            x_batch_sig, x_batch_sig_w, y_batch_sig, s_batch_sig = next(iter_any_sig)
            x_batch_bkg, x_batch_bkg_w, y_batch_bkg, s_batch_bkg = next(iter_any_bkg)
            x_batch_any   = np.concatenate( (x_batch_sig, x_batch_bkg), axis = 0)
            x_batch_any_w = np.concatenate( (x_batch_sig_w, x_batch_bkg_w), axis = 0)
            y_batch_any   = np.concatenate( (y_batch_sig, y_batch_bkg), axis = 0)
            s_batch_any   = np.concatenate( (s_batch_sig, s_batch_bkg), axis = 0)
            self.disc_fixed_adv.train_on_batch([x_batch_any, y_batch_any],
                                               [s_batch_any],
                                               sample_weight = [x_batch_any_w])

          # step generator
          x_batch_any, x_batch_any_w, y_batch_any, s_batch_any = next(iter_any)

          self.disc.trainable = True
          self.adv.trainable = False
          self.disc_adv_fixed.train_on_batch([x_batch_any, x_batch_any, y_batch_any],
                                             [y_batch_any, s_batch_any],
                                             sample_weight = [x_batch_any_w, x_batch_any_w])
  
      if epoch % self.n_eval == 0:
        disc_metric = 0
        adv_metric_nom = 0
        adv_metric_syst = 0
        x,w,y,s = next(iter_test_nom)
        disc_metric += self.disc.evaluate(x.values, y.values, sample_weight = w.values, verbose = 0)
        x,w,y,s = next(iter_test_any)
        adv_metric = self.adv.evaluate([y.values, self.disc.predict(x.values, verbose = 0)], s.values, sample_weight = w.values, verbose = 0)
        #adv_metric = self.adv.evaluate(self.disc.predict(x.values, verbose = 0), s.values, sample_weight = w.values, verbose = 0)
        x,w,y,s = next(iter_test_nom)
        adv_metric_nom += self.adv.evaluate([y.values, self.disc.predict(x.values, verbose = 0)], s.values, sample_weight = w.values, verbose = 0)
        #adv_metric_nom += self.adv.evaluate(self.disc.predict(x.values, verbose = 0), s.values, sample_weight = w.values, verbose = 0)
        x,w,y,s = next(iter_test_sys)
        adv_metric_syst += self.adv.evaluate([y.values, self.disc.predict(x.values, verbose = 0)], s.values, sample_weight = w.values, verbose = 0)
        #adv_metric_syst += self.adv.evaluate(self.disc.predict(x.values, verbose = 0), s.values, sample_weight = w.values, verbose = 0)
        if adv_metric == 0: adv_metric = 1e-20

        self.adv_loss_train = np.append(self.adv_loss_train, [adv_metric])
        self.adv_loss_nom_train = np.append(self.adv_loss_nom_train, [adv_metric_nom])
        self.adv_loss_sys_train = np.append(self.adv_loss_sys_train, [adv_metric_syst])
        self.disc_loss_train = np.append(self.disc_loss_train, [disc_metric])
        floss = h5py.File('%s/%s_loss.h5' % (result_dir, prefix), 'w')
        floss.create_dataset('adv_loss', data = self.adv_loss_train)
        floss.create_dataset('adv_loss_nom', data = self.adv_loss_nom_train)
        floss.create_dataset('adv_loss_sys', data = self.adv_loss_sys_train)
        floss.create_dataset('disc_loss', data = self.disc_loss_train)
        floss.close()

        print("Batch %5d: L_{disc. only} = %10.7f; L_{adv.} = %10.7f ; L_{adv.,nom} = %10.7f ; L_{adv.,sys} = %10.7f" % (epoch, disc_metric, adv_metric, adv_metric_nom, adv_metric_syst))
        self.save("%s/%s_discriminator_%d" % (network_dir, prefix, epoch), "%s/%s_adv_%d" % (network_dir, prefix, epoch))
      #gc.collect()

    print("============ End of training ===============")

  def load_loss(self, filename):
    floss = h5py.File(filename)
    self.adv_loss_train = floss['adv_loss'][:]
    self.adv_loss_nom_train = floss['adv_loss_nom'][:]
    self.adv_loss_sys_train = floss['adv_loss_sys'][:]
    self.disc_loss_train = floss['disc_loss'][:]
    self.n_iteration = len(self.disc_loss_train)*self.n_eval
    floss.close()

  def plot_train_metrics(self, filename, nnTaken = -1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    it = np.arange(0, self.n_iteration, self.n_eval)
    plt.plot(it, (self.disc_loss_train), linestyle = '-', color = 'r', label = r'$\mathcal{L}_{\mathrm{disc}}$')
    plt.plot(it, (np.fabs(self.adv_loss_train)), linestyle = '-', color = 'b', label = r' | $ \mathcal{L}_{\mathrm{adv}} |$')
    plt.plot(it, (np.fabs(self.disc_loss_train - self.lambda_decorr*np.fabs(self.adv_loss_train))), linestyle = '-', color = 'k', label = r'$\mathcal{L}_{\mathrm{disc}} - \lambda_{\mathrm{decorr}} |\mathcal{L}_{\mathrm{adv}}$|')
    if self.n_pretrain > 0:
      plt.axvline(x = self.n_pretrain, color = 'k', linestyle = '--', label = 'End of discriminator bootstrap')
      #plt.axvline(x = 2*self.n_pretrain, color = 'k', linestyle = ':', label = 'End of adv bootstrap')
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([1e-1, 1.0])
    ax.set_yscale('log')
    plt.legend(frameon = False)
    plt.savefig(filename)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fac = 1.0
    if np.max(np.abs(self.adv_loss_nom_train)) > 0:
      fac /= np.max(np.abs(self.adv_loss_nom_train))
    plt.plot(it, (fac*self.adv_loss_nom_train), linestyle = '-', color = 'g', label = r' $ %4.2f \mathcal{L}_{\mathrm{adv,nom}}$' % (fac) )
    plt.plot(it, (-fac*self.adv_loss_sys_train), linestyle = '-', color = 'c', label = r' $ %4.2f \mathcal{L}_{\mathrm{adv,sys}}$' % (-fac) )
    if self.n_pretrain > 0:
      plt.axvline(x = self.n_pretrain, color = 'k', linestyle = '--', label = 'End of discriminator bootstrap')
      #plt.axvline(x = 2*self.n_pretrain, color = 'k', linestyle = ':', label = 'End of adv bootstrap')
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([-1, 1])
    plt.legend(frameon = False)
    filename_adv = filename.replace('.pdf', '_adv_split.pdf')
    plt.savefig(filename_adv)
    plt.close(fig)
  
  def save(self, disc_filename, adv_filename):
    adv_json = self.adv.to_json()
    with open("%s.json" % adv_filename, "w") as json_file:
      json_file.write(adv_json)
    self.adv.save_weights("%s.h5" % adv_filename)

    disc_json = self.disc.to_json()
    with open("%s.json" % disc_filename, "w") as json_file:
      json_file.write(disc_json)
    self.disc.save_weights("%s.h5" % disc_filename)

  '''
  Load stored network
  '''
  def load(self, disc_filename, adv_filename):
    json_file = open('%s.json' % disc_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.disc = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
    self.disc.load_weights("%s.h5" % disc_filename)

    json_file = open('%s.json' % adv_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.adv = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.adv.load_weights("%s.h5" % adv_filename)

    self.adv_input = K.layers.Input(shape = (1,), name = 'adv_input')
    self.disc_input = K.layers.Input(shape = (self.n_dimensions,), name = 'disc_input')

    self.disc.compile(loss = K.losses.binary_crossentropy, optimizer = K.optimizers.Adam(lr = 1e-3), metrics = [])
    self.adv.compile(loss = K.losses.binary_crossentropy,
                        optimizer = K.optimizers.Adam(lr = 1e-3), metrics = [])
    self.create_networks()

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Train a Wasserstein GAN with gradient penalty to classify signal versus background while being insensitive to systematic variations.')
  parser.add_argument('--network-dir', dest='network_dir', action='store',
                    default='network',
                    help='Directory where networks are saved during training. (default: "network")')
  parser.add_argument('--result-dir', dest='result_dir', action='store',
                    default='result',
                    help='Directory where results are saved. (default: "result")')
  parser.add_argument('--input-file', dest='input', action='store',
                    default='input.h5',
                    help='Name of the file from where to read the input. (default: "input.h5")')
  parser.add_argument('--iterations', dest='iterations', action='store',
                    default='10001',
                    help='Number of iterations for training. (default: "10001")')
  parser.add_argument('--load-trained', dest='trained', action='store',
                    default='10000',
                    help='Number to be appended to end of filename when loading pretrained networks. Ignored during the "train" mode. (default: "10000")')
  parser.add_argument('--prefix', dest='prefix', action='store',
                    default='gan',
                    help='Prefix to be added to filenames when producing plots. (default: "gan")')
  parser.add_argument('--lambda', dest='l', action='store',
                    default=50.0,
                    help='Value of lambda_decorr. (default: 50.0)')
  parser.add_argument('--no-adv', dest='no_adv', action='store_true',
                    default=False,
                    help='De-activate adversarial training?')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_loss', 'plot_input', 'plot_output', 'plot_disc', 'plot_adv'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_loss" (plot loss from training), "plot_input" (plot input variables and correlations), "plot_output" (plot output variables), "plot_disc" (plot discriminator output), "plot_adv" (plot adv. output). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix
  trained = args.trained

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  if not os.path.exists(args.network_dir):
    os.makedirs(args.network_dir)

  if not args.no_adv:
    print("Using adversary.")
  if args.no_adv:
    print("NOT using adversary.")
  network = GAN(no_adv = args.no_adv, lambda_decorr = float(args.l), n_iteration = int(args.iterations))

  # read it from disk
  network.read_input_from_files(filename = args.input)

  var = list(network.file['df'].columns.values)
  var.remove('syst')
  var.remove('sample')
  var.remove('weight')
  var.remove('train')

  # when training make some debug plots and prepare the network
  if args.mode == 'train':
    print("Plotting correlations.")
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    print("Plotting scatter plots.")
    network.plot_scatter_input(var[0], var[1], "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))

    # create network
    network.create_networks()

    # for comparison: make a plot of the NN output value before any training
    # this will just be random!
    # try to predict if the signal or bkg. events in the test set are really signal or bkg.
    print("Plotting discriminator output.")
    network.plot_discriminator_output("%s/%s_discriminator_output_before_training.pdf" % (args.result_dir, prefix))
    print("Plotting adv output.")
    network.plot_adv_output("%s/%s_adv_output_before_training.pdf" % (args.result_dir, prefix))

    # train it
    print("Training.")
    network.train(prefix, args.result_dir, args.network_dir)

    # plot training evolution
    print("Plotting train metrics.")
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix))

    print("Plotting discriminator output after training.")
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
    print("Plotting adv output after training.")
    network.plot_adv_output("%s/%s_adv_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_loss':
    network.load_loss("%s/%s_loss.h5" % (args.result_dir, prefix))
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix), int(trained))
  elif args.mode == 'plot_disc':
    network.load("%s/%s_discriminator_%s" % (args.network_dir, prefix, trained), "%s/%s_adv_%s" % (args.network_dir, prefix, trained))
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_adv':
    network.load("%s/%s_discriminator_%s" % (args.network_dir, prefix, trained), "%s/%s_adv_%s" % (args.network_dir, prefix, trained))
    network.plot_adv_output("%s/%s_adv_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_input':
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    network.plot_scatter_input(var[0], var[1], "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))
  elif args.mode == 'plot_output':
    network.load("%s/%s_discriminator_%s" % (args.network_dir, prefix, trained), "%s/%s_adv_%s" % (args.network_dir, prefix, trained))
    network.plot_discriminator_output_syst("%s/%s_discriminator_output_syst.pdf" % (args.result_dir, prefix))
  else:
    print('Option mode not understood.')

if __name__ == '__main__':
  main()

