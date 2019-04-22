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
  N = 10
  box = np.ones(N)/float(N)
  return np.convolve(y, box, mode = 'same')

def gradient_penalty_loss(y_true, y_pred, critic, disc, input1, input2, N):
  d1 = disc(input1)
  d2 = disc(input2)
  diff = d2 - d1
  epsilon = K.backend.random_uniform_variable(shape=[N, 1], low = 0., high = 1.)
  interp_input = d1 + (epsilon*diff)
  gradients = K.backend.gradients(critic(interp_input), [interp_input])[0]
  ## not needed as there is a single element in interp_input here (the discriminator output)
  ## the only dimension left is the batch, which we just average over in the last step
  slopes = K.backend.sqrt(1e-6 + K.backend.sum(K.backend.square(gradients), axis = [1]))
  gp = K.backend.mean(K.backend.square(1 - slopes))
  return gp

def wasserstein_loss(y_true, y_pred):
  return K.backend.mean(y_true*y_pred, axis = 0)

class WGANGP(object):
  '''
  Implementation of the Wasserstein GAN with Gradient Penalty algorithm to decorrelate a discriminator in a variable S.
  Ref.: https://arxiv.org/pdf/1704.00028.pdf
  Ref.: https://arxiv.org/abs/1701.07875
  Ref.: https://arxiv.org/abs/1406.2661
  The objective of the discriminator is to separate signal (Y = 1) from backgrond (Y = 0). The critic punishes the discriminator
  if it can guess whether the sample is the nominal (N = 1) or a systematic uncertainty (N = 0).

  The discriminator outputs o = D(x), trying to minimize the cross-entropy for the signal/background classification:
  L_{disc. only} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ]

  The critic estimates a function C(o), where o = D(x), such that the Wasserstein distance between a nominal and uncertainty sample
  in the output of the discriminator can be measured as:
  W(nominal, uncertainty) = E_{nominal}[C(D(x))] - E_{uncertainty}[C(D(x))]

  One needs to impose that the function C(.) is Lipschitz, so that W(.) actually implements the Wasserstein distance, according to
  the Katorovich-Rubinstein duality. The original Wasserstein paper imposes it by restricting the critic network weights to be close to zero.
  Here, the gradient penalty is used, on which we impose that the norm of the gradient of the critic function is close to 1 everywhere
  in the line that connects nominal and uncertainty samples (it should be everywhere though.

  To impose those restrictions and punish the discriminator for leaning about nominal or systematic uncertainties (to decorrelate it),
  the following procedure is implemented:
  1) Pre-train the discriminator n_pretrain batches, so that it can separate signal from background using this loss function:
  L_{disc. only} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ]

  2) Train the critic, fixing the discriminator, in n_critic batches to minimize the Wasserstein distance
     between nominal and uncertainty in the output of the discriminator, respecting the gradient penalty condition.
     Both signal and background samples are used here.
     epsilon = batch_size samples of a uniform distribution between 0 and 1
     o_{itp} = epsilon x_{nominal} + (1 - epsilon) x_{uncertainty}
  L_{critic} = \lambda_{decorr} { E_{nominal} [ C(D(x)) ] - E_{uncertainty} [ C(D(x)) ] } + \lambda_{GP} [ 1 - || grad_{o_{itp}} C(o_{itp}) || ]^2

  3) Train the discriminator, fixing the critic, in one batch to minimize simultaneously the discriminator cross-entropy
     and to move in the direction of - grad W = E [grad_{discriminator weights} C(D(X)) ] (Theorem 3 of the Wasserstein paper).
  L_{discriminator} = E_{signal} [ - log(D(x)) ] + E_{bkg} [ - log(1 - D(x)) ] - \lambda_{decorr} { E_{nominal} [ C(D(x)) ] - E_{uncertainty} [ C(D(x)) ] }

  4) Go back to 2 and repeat this n_iteration times.
  '''

  def __init__(self, n_iteration = 1050, n_pretrain = 0, n_critic = 5,
               n_batch = 32,
               lambda_decorr = 1.0,
               lambda_gp = 10.0,
               n_eval = 50,
               no_critic = False):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_pretrain: Number of batches to run over to pre-train the discriminator.
    :param n_critic: Number of batches to train the critic on per batch of the discriminator.
    :param n_batch: Number of samples in a batch.
    :param lambda_decorr: Lambda parameter used to weigh the decorrelation term of the discriminator loss function.
    :param lambda_gp: Lambda parameter to weight the gradient penalty of the critic loss function.
    :param n_eval: Number of batches to train before evaluating metrics.
    :param no_critic: Do not train the critic, so that we can check the impact of the training in an independent discriminator.
    '''
    self.n_iteration = n_iteration
    self.n_pretrain = n_pretrain
    self.n_critic = n_critic
    self.n_batch = n_batch
    self.lambda_decorr = lambda_decorr
    self.lambda_gp = lambda_gp
    self.n_eval = n_eval
    self.no_critic = no_critic
    self.critic = None
    self.disc = None

  '''
    Create critic network.
  '''
  def create_critic(self):
    self.critic_input = Input(shape = (1,), name = 'critic_input')
    xc = self.critic_input
    xc = Dense(200, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = Dense(100, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = LayerNormalization()(xc)
    xc = Dense(50, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = LayerNormalization()(xc)
    xc = Dense(40, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = LayerNormalization()(xc)
    xc = Dense(10, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = Dense(1, activation = None)(xc)
    self.critic = Model(self.critic_input, xc, name = "critic")
    self.critic.trainable = True
    self.critic.compile(loss = wasserstein_loss,
                        optimizer = Adam(lr = 1e-3), metrics = [])

  '''
  Create discriminator network.
  '''
  def create_disc(self):
    self.disc_input = Input(shape = (self.n_dimensions,), name = 'disc_input')

    xd = self.disc_input
    xd = Dense(200, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = Dense(100, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = LayerNormalization()(xd)
    xd = Dense(50, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = LayerNormalization()(xd)
    xd = Dense(40, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = LayerNormalization()(xd)
    xd = Dense(30, activation = None)(xd)
    xd = K.layers.LeakyReLU(0.2)(xd)
    xd = LayerNormalization()(xd)
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
    if not self.critic:
      self.create_critic()
    if not self.disc:
      self.create_disc()

    self.disc.trainable = False
    self.critic.trainable = True

    self.dummy_input = Input(shape = (1,), name = 'dummy_input')
    self.nominal_input = Input(shape = (self.n_dimensions,), name = 'nominal_input')
    self.syst_input = Input(shape = (self.n_dimensions,), name = 'syst_input')
    self.nominal_input_w = Input(shape = (1,), name = 'nominal_input_w')
    self.syst_input_w = Input(shape = (1,), name = 'syst_input_w')

    from functools import partial
    partial_gp_loss = partial(gradient_penalty_loss, critic = self.critic, disc = self.disc, input1 = self.nominal_input, input2 = self.syst_input, N = self.n_batch)

    wdistance = K.layers.Subtract()([K.layers.Multiply()([self.critic(self.disc(self.nominal_input)), self.nominal_input_w]),
                                     K.layers.Multiply()([self.critic(self.disc(self.syst_input)), self.syst_input_w])])
    wdistance_nom = K.layers.Multiply()([self.critic(self.disc(self.nominal_input)), self.nominal_input_w])

    self.disc_fixed_critic = Model([self.nominal_input, self.syst_input, self.dummy_input, self.nominal_input_w, self.syst_input_w],
                                   [wdistance, self.dummy_input],
                                   name = "disc_fixed_critic")
    self.disc_fixed_critic.compile(loss = [wasserstein_loss, partial_gp_loss],
                                   loss_weights = [self.lambda_decorr, self.lambda_gp],
                                   optimizer = Adam(lr = 5e-5, beta_1 = 0, beta_2 = 0.9), metrics = [])

    self.disc.trainable = True
    self.critic.trainable = False
    self.disc_critic_fixed = Model([self.disc_input, self.nominal_input, self.syst_input, self.nominal_input_w, self.syst_input_w],
                                   [self.disc(self.disc_input), wdistance],
                                   name = "disc_critic_fixed")
    self.disc_critic_fixed.compile(loss = [K.losses.binary_crossentropy, wasserstein_loss],
                                   loss_weights = [1.0, -self.lambda_decorr],
                                   optimizer = Adam(lr = 5e-5, beta_1 = 0, beta_2 = 0.9), metrics = [])


    print("Signal/background discriminator:")
    self.disc.trainable = True
    self.disc.summary()
    print("Critic:")
    self.critic.trainable = True
    self.critic.summary()
    print("Disc. against critic:")
    self.disc.trainable = True
    self.critic.trainable = False
    self.disc_critic_fixed.summary()
    print("Critic against disc.:")
    self.disc.trainable = False
    self.critic.trainable = True
    self.disc_fixed_critic.summary()


  '''
  '''
  def read_input_from_files(self, filename = 'input_preprocessed.h5'):
    self.file = pd.HDFStore(filename, 'r')
    self.n_dimensions = self.file['df'].shape[1]-3
    self.col_signal = self.file['df'].columns.get_loc('sample')
    self.col_syst = self.file['df'].columns.get_loc('syst')
    self.col_weight = self.file['df'].columns.get_loc('weight')

  '''
  Generate test sample.
  :param adjust_signal_weights: If True, weights the signal by the ratio of signal to background weights, so that the training considers both equally.
  '''
  def prepare_input(self, filename = 'input_preprocessed.h5', adjust_signal_weights = True, set_unit_weights = True):
    # make input file
    N = 10000
    self.file = pd.HDFStore(filename, 'w')
    x = {}
    all_data = np.zeros(shape = (0, 3+2))
    for t in ['train', 'test']:
      for s in [0, 1]:
        signal = np.random.normal(loc = -1.0 + s*0.1, scale = 0.5 + s*0.1, size = (N, 2))
        bkg    = np.random.normal(loc =  1.0 - s*0.1, scale = 0.5 - s*0.1, size = (N, 2))
        data   = np.append(signal, bkg, axis = 0)
        data_t = np.append(np.ones(N), np.zeros(N))
        data_w = np.append(np.ones(N), np.ones(N))
        data_s = s*np.append(np.ones(N), np.ones(N))
        add_all_data = np.concatenate( (data_t[:,np.newaxis], data_s[:, np.newaxis], data_w[:,np.newaxis], data), axis=1)
        all_data = np.concatenate((all_data, add_all_data), axis = 0)
      print('Checking nans in %s' % t)
      self.check_nans(all_data)
    df = pd.DataFrame(all_data, columns = ['sample', 'syst', 'weight', 'A', 'B'])
    self.file.put('df', df, format = 'table', data_columns = True)

    for t in ['train', 'test']:
      if t == 'train':
        part = pd.DataFrame( (df.index < 4*N) )
        bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index < 4*N)))
        sig = pd.DataFrame( ((df['sample'] == 1) & (df.index < 4*N)))
        syst = pd.DataFrame( ((df['syst'] == 1) & (df.index < 4*N)))
        nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index < 4*N)))
      else:
        part = pd.DataFrame( (df.index >= 4*N) )
        bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index >= 4*N)))
        sig = pd.DataFrame( ((df['sample'] == 1) & (df.index >= 4*N)))
        syst = pd.DataFrame( ((df['syst'] == 1) & (df.index >= 4*N)))
        nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index >= 4*N)))
      self.file.put('%s' % t, part, format = 'table')
      self.file.put('%s_bkg' % t, bkg, format = 'table')
      self.file.put('%s_sig'% t, sig, format = 'table')
      self.file.put('%s_syst' % t, syst, format = 'table')
      self.file.put('%s_nominal' % t, nominal, format = 'table')

    self.file.close()

  def check_nans(self, x):
    print("Dump of NaNs:")
    nan_idx = np.where(np.isnan(x))
    print(x[nan_idx])

    assert len(x[nan_idx]) == 0

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
    x = self.file['df'][nominal].loc[:, (var1, var2)]
    y = self.file['df'][nominal].loc[:, 'sample']
    g = sns.scatterplot(x = x.loc[:, var1], y = x.loc[:, var2], hue = y,
                        hue_order = [var1, var2], markers = ["^", "v"], legend = "brief", ax = ax)
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
    for x,w,y in self.get_batch(origin = 'test', signal = True): out_signal.extend(self.disc.predict(x))
    out_signal = np.array(out_signal)
    out_bkg = []
    for x,w,y in self.get_batch(origin = 'test', signal = False): out_bkg.extend(self.disc.predict(x))
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
    for x,w,y in self.get_batch(origin = 'test', signal = True, syst = False): out_signal.extend(self.disc.predict(x))
    out_signal = np.array(out_signal)
    out_bkg = []
    for x,w,y in self.get_batch(origin = 'test', signal = False, syst = False): out_bkg.extend(self.disc.predict(x))
    out_bkg = np.array(out_bkg)

    out_signal_s = []
    for x,w,y in self.get_batch(origin = 'test', signal = True, syst = True): out_signal_s.extend(self.disc.predict(x))
    out_signal_s = np.array(out_signal_s)
    out_bkg_s = []
    for x,w,y in self.get_batch(origin = 'test', signal = False, syst = True): out_bkg_s.extend(self.disc.predict(x))
    out_bkg_s = np.array(out_bkg_s)

    Nbins = 20
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

    ax[1].set_ylim([0.0, 2.0])
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

  def plot_critic_output(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # use get_continuour_batch to read directly from the file
    out_syst_nominal = []
    for x,w,y in self.get_batch(origin = 'test', syst = False): out_syst_nominal.extend(self.critic.predict(self.disc.predict(x)))
    out_syst_nominal = np.array(out_syst_nominal)
    out_syst_var = []
    for x,w,y in self.get_batch(origin = 'test', syst = True): out_syst_var.extend(self.critic.predict(self.disc.predict(x)))
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
        x_batch = df.drop(['weight', 'sample', 'syst'], axis = 1)
        x_batch_w = df.loc[:, 'weight']
        y_batch = df.loc[:, 'sample']
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
        x_batch = df.drop(['weight', 'sample', 'syst'], axis = 1)
        x_batch_w = df.loc[:, 'weight']
        y_batch = df.loc[:, 'sample']
        s_batch = df.loc[:, 'syst']
        yield x_batch, x_batch_w, y_batch, s_batch

  def train(self, prefix, result_dir, network_dir):
    # algorithm:
    # 0) pretrain discriminator
    # 1) Train adv. to guess syst. (freezing discriminator)
    # 2) Train disc. to fool adv. (freezing adv.)
    self.critic_gp_loss_train = np.array([])
    self.critic_loss_train = np.array([])
    self.critic_loss_nom_train = np.array([])
    self.critic_loss_sys_train = np.array([])
    self.disc_loss_train = np.array([])
    positive_y = np.ones(self.n_batch)
    iter_nom = self.get_batch(origin = 'train', syst = False, noStop = True)
    iter_sys = self.get_batch(origin = 'train', syst = True, noStop = True)
    iter_test_nom = self.get_batch(origin = 'test', syst = False, noStop = True)
    iter_test_sys = self.get_batch(origin = 'test', syst = True, noStop = True)
    for epoch in range(self.n_iteration):
      if self.no_critic:
        x_batch_nom, x_batch_nom_w, y_batch_nom, s_batch_nom = next(iter_nom)
        self.disc.trainable = True
        self.disc.train_on_batch(x_batch_nom, y_batch_nom, sample_weight = x_batch_nom_w)

      if not self.no_critic:
        # step 0 - pretraining
        if epoch < self.n_pretrain:
          x_batch_nom, x_batch_nom_w, y_batch_nom, s_batch_nom = next(iter_nom)
          self.disc.trainable = True
          self.disc.train_on_batch(x_batch_nom, y_batch_nom, sample_weight = x_batch_nom_w) # reconstruct input in auto-encoder

        if epoch >= self.n_pretrain:
          # step critic
          n_critic = self.n_critic
          for k in range(0, n_critic):
            x_batch_nom, x_batch_nom_w, y_batch_nom, s_batch_nom = next(iter_nom)
            x_batch_syst, x_batch_syst_w, y_batch_syst, s_batch_syst = next(iter_sys)

            self.disc.trainable = False
            self.critic.trainable = True
            self.disc_fixed_critic.train_on_batch([x_batch_nom, x_batch_syst, positive_y, x_batch_nom_w, x_batch_syst_w],
                                                  [positive_y, positive_y],
                                                  sample_weight = [positive_y, positive_y])

          # step generator
          x_batch_nom, x_batch_nom_w, y_batch_nom, s_batch_nom = next(iter_nom)
          x_batch_syst, x_batch_syst_w, y_batch_syst, s_batch_syst = next(iter_sys)

          self.disc.trainable = True
          self.critic.trainable = False
          self.disc_critic_fixed.train_on_batch([x_batch_nom, x_batch_nom, x_batch_syst, x_batch_nom_w, x_batch_syst_w],
                                                [y_batch_nom, positive_y],
                                                sample_weight = [x_batch_nom_w, positive_y])
  
      if epoch % self.n_eval == 0:
        disc_metric = 0
        critic_metric_nom = 0
        critic_metric_syst = 0
        x,w,y,s = next(iter_test_nom)
        disc_metric += self.disc.evaluate(x, y, sample_weight = w, verbose = 0)
        if epoch >= self.n_pretrain and not self.no_critic:
          critic_metric_nom += np.sum(self.critic.predict(self.disc.predict(x, verbose = 0))*w)/np.sum(w)
        if epoch >= self.n_pretrain and not self.no_critic:
          x,w,y,s = next(iter_test_sys)
          critic_metric_syst += np.sum(self.critic.predict(self.disc.predict(x, verbose = 0))*w)/np.sum(w)
        critic_metric = critic_metric_nom - critic_metric_syst
        if critic_metric == 0: critic_metric = 1e-20

        critic_gradient_penalty = 0
        if epoch >= self.n_pretrain and not self.no_critic:
          x,w,y,s = next(iter_test_nom)
          xs,ws,ys,ss = next(iter_test_sys)

          self.disc.trainable = False
          self.critic.trainable = True
          critic_gradient_penalty += self.disc_fixed_critic.evaluate([x, xs, positive_y, w, ws],
                                                                     [positive_y, positive_y],
                                                                     sample_weight = [positive_y, positive_y], verbose = 0)[-1]
        if critic_gradient_penalty == 0: critic_gradient_penalty = 1e-20

        self.critic_loss_train = np.append(self.critic_loss_train, [critic_metric])
        self.critic_loss_nom_train = np.append(self.critic_loss_nom_train, [critic_metric_nom])
        self.critic_loss_sys_train = np.append(self.critic_loss_sys_train, [critic_metric_syst])
        self.disc_loss_train = np.append(self.disc_loss_train, [disc_metric])
        self.critic_gp_loss_train = np.append(self.critic_gp_loss_train, [critic_gradient_penalty])
        floss = h5py.File('%s/%s_loss.h5' % (result_dir, prefix), 'w')
        floss.create_dataset('critic_loss', data = self.critic_loss_train)
        floss.create_dataset('critic_loss_nom', data = self.critic_loss_nom_train)
        floss.create_dataset('critic_loss_sys', data = self.critic_loss_sys_train)
        floss.create_dataset('disc_loss', data = self.disc_loss_train)
        floss.create_dataset('critic_gp_loss', data = self.critic_gp_loss_train)
        floss.close()

        print("Batch %5d: L_{disc. only} = %10.7f; - lambda_{decorr} L_{critic} = %10.7f ; L_{critic,nom} = %10.7f ; L_{critic,sys} = %10.7f ; lambda_{gp} (|grad C| - 1)^2 = %10.7f" % (epoch, disc_metric, -self.lambda_decorr*critic_metric, critic_metric_nom, critic_metric_syst, self.lambda_gp*critic_gradient_penalty))
        self.save("%s/%s_disc_%d" % (network_dir, prefix, epoch), "%s/%s_critic_%d" % (network_dir, prefix, epoch))
      #gc.collect()

    print("============ End of training ===============")

  def load_loss(self, filename):
    floss = h5py.File(filename)
    self.critic_loss_train = floss['critic_loss'][:]
    self.critic_loss_nom_train = floss['critic_loss_nom'][:]
    self.critic_loss_sys_train = floss['critic_loss_sys'][:]
    self.disc_loss_train = floss['disc_loss'][:]
    self.critic_gp_loss_train = floss['critic_gp_loss'][:]
    self.n_iteration = len(self.disc_loss_train)*self.n_eval
    floss.close()

  def plot_train_metrics(self, filename, nnTaken = -1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    it = np.arange(0, self.n_iteration, self.n_eval)
    plt.plot(it, (self.disc_loss_train), linestyle = '-', color = 'r', label = r'$\mathcal{L}_{\mathrm{disc}}$')
    plt.plot(it, (np.fabs(-self.lambda_decorr*self.critic_loss_train)), linestyle = '-', color = 'b', label = r' | $\lambda_{\mathrm{decorr}} \mathcal{L}_{\mathrm{critic}} |$')
    plt.plot(it, (self.lambda_gp*self.critic_gp_loss_train), linestyle = '-', color = 'grey', label = r'$\lambda_{\mathrm{gp}} (||\nabla_{\hat{x}} C(\hat{x})||_{2} - 1)^2$')
    plt.plot(it, (np.abs(self.disc_loss_train - self.lambda_decorr*np.abs(self.critic_loss_train))), linestyle = '-', color = 'k', label = r'$\mathcal{L}_{\mathrm{disc}} - \lambda_{\mathrm{decorr}} |\mathcal{L}_{\mathrm{critic}}$|')
    if self.n_pretrain > 0:
      plt.axvline(x = self.n_pretrain, color = 'k', linestyle = '--', label = 'End of discriminator bootstrap')
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([1e-3, 10])
    ax.set_yscale('log')
    plt.legend(frameon = False)
    plt.savefig(filename)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fac = 1.0
    if np.max(np.abs(self.critic_loss_nom_train)) > 0:
      fac /= np.max(np.abs(self.critic_loss_nom_train))
    plt.plot(it, (fac*self.critic_loss_nom_train), linestyle = ':', color = 'g')
    plt.plot(it, (-fac*self.critic_loss_sys_train), linestyle = ':', color = 'c')
    plt.plot(it, smoothen(fac*self.critic_loss_nom_train), linestyle = '-', color = 'g', label = r' $ %4.2f \mathcal{L}_{\mathrm{critic,nom}}$' % (fac) )
    plt.plot(it, smoothen(-fac*self.critic_loss_sys_train), linestyle = '-', color = 'c', label = r' $ %4.2f \mathcal{L}_{\mathrm{critic,sys}}$' % (-fac) )
    if self.n_pretrain > 0:
      plt.axvline(x = self.n_pretrain, color = 'k', linestyle = '--', label = 'End of discriminator bootstrap')
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([-1, 1])
    plt.legend(frameon = False)
    filename_crit = filename.replace('.pdf', '_critic_split.pdf')
    plt.savefig(filename_crit)
    plt.close(fig)
  
  def save(self, disc_filename, critic_filename):
    critic_json = self.critic.to_json()
    with open("%s.json" % critic_filename, "w") as json_file:
      json_file.write(critic_json)
    self.critic.save_weights("%s.h5" % critic_filename)

    disc_json = self.disc.to_json()
    with open("%s.json" % disc_filename, "w") as json_file:
      json_file.write(disc_json)
    self.disc.save_weights("%s.h5" % disc_filename)

  '''
  Load stored network
  '''
  def load(self, disc_filename, critic_filename):
    json_file = open('%s.json' % disc_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.disc = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
    self.disc.load_weights("%s.h5" % disc_filename)

    json_file = open('%s.json' % critic_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.critic = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.critic.load_weights("%s.h5" % critic_filename)

    self.critic_input = K.layers.Input(shape = (1,), name = 'critic_input')
    self.disc_input = K.layers.Input(shape = (self.n_dimensions,), name = 'disc_input')

    self.disc.compile(loss = K.losses.binary_crossentropy, optimizer = K.optimizers.Adam(lr = 1e-3), metrics = [])
    self.critic.compile(loss = wasserstein_loss,
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
                    help='Name of the file from where to read the input. If the file does not exist, create it. (default: "input.h5")')
  parser.add_argument('--load-trained', dest='trained', action='store',
                    default='1000',
                    help='Number to be appended to end of filename when loading pretrained networks. Ignored during the "train" mode. (default: "1950")')
  parser.add_argument('--prefix', dest='prefix', action='store',
                    default='wgangp',
                    help='Prefix to be added to filenames when producing plots. (default: "wgangp")')
  parser.add_argument('--no-critic', dest='no_critic', action='store',
                    default=False,
                    help='If True, train only the discriminator. (default: False)')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_loss', 'plot_input', 'plot_output', 'plot_disc', 'plot_critic'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_loss" (plot loss from training), "plot_input" (plot input variables and correlations), "plot_output" (plot output variables), "plot_disc" (plot discriminator output), "plot_critic" (plot critic output). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix
  trained = args.trained

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  if not os.path.exists(args.network_dir):
    os.makedirs(args.network_dir)

  network = WGANGP(no_critic = args.no_critic)
  # apply pre-processing if the preprocessed file does not exist
  if not os.path.isfile(args.input):
    network.prepare_input(filename = args.input)

  # read it from disk
  network.read_input_from_files(filename = args.input)

  var = list(network.file['df'].columns.values)
  var.remove('syst')
  var.remove('sample')
  var.remove('weight')

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
    print("Plotting critic output.")
    network.plot_critic_output("%s/%s_critic_output_before_training.pdf" % (args.result_dir, prefix))

    # train it
    print("Training.")
    network.train(prefix, args.result_dir, args.network_dir)

    # plot training evolution
    print("Plotting train metrics.")
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix))

    print("Plotting discriminator output after training.")
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
    print("Plotting critic output after training.")
    network.plot_critic_output("%s/%s_critic_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_loss':
    network.load_loss("%s/%s_loss.h5" % (args.result_dir, prefix))
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix), int(trained))
  elif args.mode == 'plot_disc':
    network.load("%s/%s_disc_%s" % (args.network_dir, prefix, trained), "%s/%s_critic_%s" % (args.network_dir, prefix, trained))
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_critic':
    network.load("%s/%s_disc_%s" % (args.network_dir, prefix, trained), "%s/%s_critic_%s" % (args.network_dir, prefix, trained))
    network.plot_critic_output("%s/%s_critic_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_input':
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    network.plot_scatter_input(var[0], var[1], "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))
  elif args.mode == 'plot_output':
    network.load("%s/%s_disc_%s" % (args.network_dir, prefix, trained), "%s/%s_critic_%s" % (args.network_dir, prefix, trained))
    network.plot_discriminator_output_syst("%s/%s_discriminator_output_syst.pdf" % (args.result_dir, prefix))
  else:
    print('Option mode not understood.')

if __name__ == '__main__':
  main()

