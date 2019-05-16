#!/usr/bin/env python3

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
import sklearn
import copy

names = ['A', 'B']

'''
Generate a toy sample for signal and background.
'''
def make_sample(syst, N):
  global names
  data_o, data_t = sklearn.datasets.make_moons(n_samples = N, noise = 0.3)
  data = copy.deepcopy(data_o)
  theta = 0
  if syst > 0.5:
    theta = 10*np.pi/180.
  data[:,0] = np.cos(theta)*data_o[:,0] - np.sin(theta)*data[:,1] 
  data[:,1] = np.sin(theta)*data_o[:,0] + np.cos(theta)*data[:,1] 
  return data, data_t, names

'''
Generate toy sample and save it on disk.
'''
def prepare_input(filename = 'input_ee.h5'):
  global names
  import sklearn.datasets
  # make input file
  N = 20000
  file = pd.HDFStore(filename, 'w')
  x = {}
  NumberOfVars = len(names)
  all_data = np.zeros(shape = (0, 4+NumberOfVars))
  for t in ['train', 'test']:
    for s in [0, 1]:
      data, data_t, names = make_sample(syst = s, N = N)
      data_tr = np.ones(N)
      data_w = np.ones(N)
      data_s = s*np.ones(N)
      add_all_data = np.concatenate( (data_t[:,np.newaxis], data_s[:, np.newaxis], data_w[:,np.newaxis], data_tr[:, np.newaxis], data), axis=1)
      all_data = np.concatenate((all_data, add_all_data), axis = 0)
    print('Checking nans in %s' % t)
    check_nans(all_data)
  df = pd.DataFrame(all_data, columns = ['sample', 'syst', 'weight', 'train']+names)
  df['train'] = (df.index < 2*N)
  file.put('df', df, format = 'table', data_columns = True)

  for t in ['train', 'test']:
    if t == 'train':
      part = pd.DataFrame( (df.index < 2*N) )
      bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index < 2*N)))
      sig = pd.DataFrame( ((df['sample'] == 1) & (df.index < 2*N)))
      syst = pd.DataFrame( ((df['syst'] == 1) & (df.index < 2*N)))
      nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index < 2*N)))
    else:
      part = pd.DataFrame( (df.index >= 2*N) )
      bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index >= 2*N)))
      sig = pd.DataFrame( ((df['sample'] == 1) & (df.index >= 2*N)))
      syst = pd.DataFrame( ((df['syst'] == 1) & (df.index >= 2*N)))
      nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index >= 2*N)))
    file.put('%s' % t, part, format = 'table')
    file.put('%s_bkg' % t, bkg, format = 'table')
    file.put('%s_sig'% t, sig, format = 'table')
    file.put('%s_syst' % t, syst, format = 'table')
    file.put('%s_nominal' % t, nominal, format = 'table')

  file.close()

def check_nans(x):
  print("Dump of NaNs:")
  nan_idx = np.where(np.isnan(x))
  print(x[nan_idx])

  assert len(x[nan_idx]) == 0

def plotRatio(filename = 'input_ee.h5', num = "Variation", den = "Nominal", var = "diHiggsPt"):
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import copy
  df = pd.read_hdf(filename, 'df')
  mpl.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14,
                       'axes.titlesize': 14, 'axes.labelsize': 14,
                       'legend.fontsize': 14, 'font.size': 14})
  fig = plt.figure(figsize = (10, 8))
  gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
  ax1 = fig.add_subplot(gs[0])
  ax2 = fig.add_subplot(gs[1], sharex=ax1)
  plt.setp(ax1.get_xticklabels(), visible=False)
  fig.subplots_adjust(hspace=0.001)
  #fig, (ax1, ax2) = plt.subplots(nrows=2)
  ns = []
  ns_err = []
  cutNum = (df['syst'] == 1)
  cutDen = (df['syst'] == 0)

  # find bin size
  c, bins = np.histogram(df.loc[cutDen, var], bins = 20, weights = df['weight'][cutDen])
  for k in range(len(bins-1)):
    if np.sum(c[0:k])/np.sum(c) > 0.99:
      bins = np.arange(bins[0], bins[k], (bins[k] - bins[0])/20)
      break
  binwidth = bins[1]-bins[0]
  s = binwidth*0.5

  cutNum = (df['syst'] == 1) & (df['sample'] == 0)
  cutDen = (df['syst'] == 0) & (df['sample'] == 0)
  c, b = np.histogram(df.loc[cutNum, var], bins = bins, weights = df['weight'][cutNum])
  ns.append(c)
  c, b = np.histogram(df.loc[cutDen, var], bins = bins, weights = df['weight'][cutDen])
  ns.append(c)
  c, b = np.histogram(df.loc[cutNum, var], bins = bins, weights = df['weight'][cutNum]**2)
  ns_err.append(np.sqrt(c))
  c, b = np.histogram(df.loc[cutDen, var], bins = bins, weights = df['weight'][cutDen]**2)
  ns_err.append(np.sqrt(c))
  
  ax1.step(bins[:-1], ns[0], alpha = 0.8, label = num + ' bkg.', color = 'r')
  ax1.step(bins[:-1], ns[1], alpha = 0.8, label = den + ' bkg.', color = 'magenta')
  ax1.errorbar(bins[:-2]+s, ns[0][1:], ns_err[0][1:], alpha = 0.8, color = 'r', fmt = 'o')
  ax1.errorbar(bins[:-2]+s, ns[1][1:], ns_err[1][1:], alpha = 0.8, color = 'magenta', fmt = 'o')

  cutNum = (df['syst'] == 1) & (df['sample'] == 1)
  cutDen = (df['syst'] == 0) & (df['sample'] == 1)
  c, b = np.histogram(df.loc[cutNum, var], bins = bins, weights = df['weight'][cutNum])
  ns.append(c)
  c, b = np.histogram(df.loc[cutDen, var], bins = bins, weights = df['weight'][cutDen])
  ns.append(c)
  c, b = np.histogram(df.loc[cutNum, var], bins = bins, weights = df['weight'][cutNum]**2)
  ns_err.append(np.sqrt(c))
  c, b = np.histogram(df.loc[cutDen, var], bins = bins, weights = df['weight'][cutDen]**2)
  ns_err.append(np.sqrt(c))
  
  ax1.step(bins[:-1], ns[2], alpha = 0.8, label = num + ' sig.', color = 'b')
  ax1.step(bins[:-1], ns[3], alpha = 0.8, label = den + ' sig.', color = 'g')
  ax1.errorbar(bins[:-2]+s, ns[2][1:], ns_err[2][1:], alpha = 0.8, color = 'b', fmt = 'o')
  ax1.errorbar(bins[:-2]+s, ns[3][1:], ns_err[3][1:], alpha = 0.8, color = 'g', fmt = 'o')

  ax1.set_ylim([0, None])
  ax1.legend()

  rat1 = copy.deepcopy(ns[0])
  rat1_err = copy.deepcopy(ns_err[0])
  rat2 = copy.deepcopy(ns[2])
  rat2_err = copy.deepcopy(ns_err[2])
  for k in range(0, len(rat1)):
    rat1[k] = 0
    rat1_err[k] = 0
    rat2[k] = 0
    rat2_err[k] = 0
    if ns[1][k] != 0 and ns[0][k] != 0:
      rat1[k] = ns[0][k]/ns[1][k]
      rat1_err[k] = rat1[k]*np.sqrt( (ns_err[0][k]/ns[0][k])**2 + (ns_err[1][k]/ns[1][k])**2 )
    if ns[3][k] != 0 and ns[2][k] != 0:
      rat2[k] = ns[2][k]/ns[3][k]
      rat2_err[k] = rat2[k]*np.sqrt( (ns_err[2][k]/ns[2][k])**2 + (ns_err[3][k]/ns[3][k])**2 )
 
  ax2.step(bins[:-1], rat1, alpha=0.8, color = 'r')
  ax2.step(bins[:-1], rat2, alpha=0.8, color = 'b')
  ax2.errorbar(bins[:-2]+s, rat1[1:], rat1_err[1:], alpha=0.8, color = 'r', fmt = 'o')
  ax2.errorbar(bins[:-2]+s, rat2[1:], rat2_err[1:], alpha=0.8, color = 'b', fmt = 'o')

  ax1.grid(linestyle = '-')
  ax2.grid(linestyle = '-')
  ax1.set_ylabel("Entries")
  ax2.set_ylabel('Ratio (%s/%s)' % (num, den))
  ax2.set_xlabel(var)
  ax2.set_ylim(0.8, 1.2)
  yticks = ax1.yaxis.get_major_ticks() 
  yticks[0].label1.set_visible(False)
  plt.savefig("generateToys_ratio_%s_%s_%s.pdf" % (var, num, den))
  plt.close("all")

if __name__ == '__main__':
  filename = 'input_toys.h5'
  prepare_input(filename)
  for var in names:
    plotRatio(filename, var = var)

