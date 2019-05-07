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

names = ['m_pp', 'pt_pp', 'pt_p1', 'pt_p2', 'theta1', 'theta2', 'phi1', 'phi2']

'''
Simulate e e > e e, assuming it is produced by an s-channel diagram with intermediate particle with mass m_res and width L.
'''
def getCrossSection(initialP, m_res, L = 1):
  # 2 -> 2 scattering. Only 2 unknowns: the angles
  a = np.random.uniform(0.1, 0.9)
  b = np.random.uniform(0., 1.)
  # angle phi sum 2*pi, since the particles are back-to-back
  phi1 = 2*np.pi*b
  phi2 = 2*np.pi*(1. - b)
  p = initialP    # energy-momentum conservation implies the same momentum in the final state
  # angle theta sum pi
  # use different random number for theta
  theta1 = np.pi*a
  theta2 = np.pi*(1. - a)
  # calculate pseudo-rapidity
  eta1 = -np.log(np.tan(theta1*0.5))
  eta2 = -eta1
  # calculate pt of particles
  pt_p1 = p*np.sin(theta1)
  pt_p2 = pt_p1
  # calculate s_hat:
  # s_hat = (p1 + p2)^mu (p1 + p2)_mu
  #       = (E1 + E2)^2 - |p1 + p2|^2
  #       = 4*p^2 - p^2 - p^2 - 2 * p1.p2
  #       = 2*p^2 - 2*p1.p2
  #       = 2*(p^2 - |p1| |p2| cos angle)
  #       = 2 p^2 (1 - cos angle)
  # angle = 180 deg
  # s_hat = 4 p^2
  s_hat = 4*(p**2)
  # transverse momentum of the final-state particles: pt_pp = (p1*sin(theta1)*cos(phi1) + p2*sin(theta2)*cos(phi2), p1*sin(theta1)*sin(phi1) + p2*sin(theta2)*sin(phi2))
  #pt_pp = np.sqrt((pt_p1*np.cos(phi1) + pt_p2*np.cos(phi2))**2 + (pt_p1*np.sin(phi1) + pt_p2*np.sin(phi2))**2)
  pt_pp = pt_p1 * np.sqrt(2 + 2*np.cos(phi1)*np.cos(phi2) + 2*np.sin(phi1)*np.sin(phi2))
  # dsigma/dOmega = 1/(64*pi^2*s) * |k1^prime|/|k1| * |M|^2
  w = 1.0
  w *= (1.0/(64.*np.pi*np.pi*s_hat)) # * |k1^prime|/|k1| = 1.0
  # finally multiply by the matrix element squared
  w *= 1.0/(np.power(s_hat - m_res**2, 2) + (m_res**2)*(L**2))
  w *= np.sin(theta1)*np.pi*2*np.pi   # sin(theta1) from dOmega, and as we integrate the cross section in a and b and not phi1 and theta1, add two factors of np.pi from the Jacobian
  return [w, pt_pp, pt_p1, pt_p2, theta1, theta2, phi1, phi2, s_hat]

'''
Generate a toy sample for signal and background.
Simulates electron-electron scattering with initial momentum of initialP [in GeV].
Signal generated under the model: prob(s hat) = 1/(s hat - m^2)^2, where m = 50. GeV
Background generated under the model: prob(s hat) = 1/(s hat)^2.
Signal generated with 50% probability relative to background.
'''
def make_sample(syst, N):
  global names
  data = np.zeros(shape = (N, len(names)))
  data_t = np.zeros(shape = (N,))
  minP = 10
  maxP = 100.
  m_res = 50.
  L = 0.05*m_res
  unc_res = 0.04*m_res
  # thermalise and get w_max
  w_sig_max = 0.
  for i in range(0, int(0.1*N)):
    initialP = np.random.uniform(minP, maxP) # initial electron momentum
    unc_mass = syst*np.random.normal(0., unc_res) # uncertainty of 10 GeV on the mass
    w, pt_pp, pt_p1, pt_p2, eta1, eta2, phi1, phi2, s_hat = getCrossSection(initialP, m_res = m_res + unc_mass, L = L)
    if w > w_sig_max:
      w_sig_max = w
  w_bkg_max = 0.
  for i in range(0, int(0.1*N)):
    initialP = np.random.uniform(minP, maxP) # initial electron momentum
    w, pt_pp, pt_p1, pt_p2, eta1, eta2, phi1, phi2, s_hat = getCrossSection(initialP, m_res = 0.)
    if w > w_bkg_max:
      w_bkg_max = w
  for i in range(0, N):
    isSignal = np.random.uniform(0., 1.)
    if isSignal > 0.5:
      isSignal = 1.0
    else:
      isSignal = 0.0
    w = 0
    # get sample and unweight
    while True:
      initialP = np.random.uniform(minP, maxP) # initial electron momentum
      r = np.random.uniform(0., 1.)
      if isSignal > 0.5:
        unc_mass = syst*np.random.normal(0., unc_res) # uncertainty of 10 GeV on the mass
        w, pt_pp, pt_p1, pt_p2, eta1, eta2, phi1, phi2, s_hat = getCrossSection(initialP, m_res = m_res + unc_mass, L = L)
        wmax = w_sig_max
      else:
        w, pt_pp, pt_p1, pt_p2, eta1, eta2, phi1, phi2, s_hat = getCrossSection(initialP, m_res = 0.)
        wmax = w_bkg_max
      if r < w/wmax:
        break
    if i % 1000 == 0:
      print("Accepted event %d / %d" % (i, N))
    data[i, 0] = np.sqrt(s_hat)
    data[i, 1] = pt_pp
    data[i, 2] = pt_p1
    data[i, 3] = pt_p2
    data[i, 4] = eta1
    data[i, 5] = eta2
    data[i, 6] = phi1
    data[i, 7] = phi2
    data_t[i] = isSignal
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
  all_data = np.zeros(shape = (0, 3+NumberOfVars))
  for t in ['train', 'test']:
    for s in [0, 1]:
      data, data_t, names = make_sample(syst = s, N = N)
      data_w = np.ones(N)
      data_s = s*np.ones(N)
      add_all_data = np.concatenate( (data_t[:,np.newaxis], data_s[:, np.newaxis], data_w[:,np.newaxis], data), axis=1)
      all_data = np.concatenate((all_data, add_all_data), axis = 0)
    print('Checking nans in %s' % t)
    check_nans(all_data)
  df = pd.DataFrame(all_data, columns = ['sample', 'syst', 'weight']+names)
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
  ax1.step(bins[:-1], ns[1], alpha = 0.8, label = den + ' bkg.', color = 'pink')
  ax1.errorbar(bins[:-2]+s, ns[0][1:], ns_err[0][1:], alpha = 0.8, color = 'r', fmt = 'o')
  ax1.errorbar(bins[:-2]+s, ns[1][1:], ns_err[1][1:], alpha = 0.8, color = 'pink', fmt = 'o')

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
  plt.savefig("produceToys_ratio_%s_%s_%s.pdf" % (var, num, den))
  plt.close("all")

if __name__ == '__main__':
  filename = 'input_ee.h5'
  prepare_input(filename)
  for var in names:
    plotRatio(filename, var = var)

