#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# to be able to run this (besides ROOT):
# sudo apt-get install python python-pip
# pip install --user matplotlib seaborn numpy pandas tensorflow keras

# use this to read the ROOT file and save it in a .h5 file to use with Pandas (faster)
def transformROOTToPandas(treeNameList = ["Nominal",
                                          'JET_JER_SINGLE_NP__1up'],
                                          #'JET_SR1_JET_GroupedNP_1__1up', 'JET_SR1_JET_GroupedNP_2__1up', 'JET_SR1_JET_GroupedNP_3__1up',
                                          #'TAUS_TRUEHADTAU_SME_TES_DETECTOR__1up', 'TAUS_TRUEHADTAU_SME_TES_INSITU__1up',
                                          #'TAUS_TRUEHADTAU_SME_TES_MODEL__1up'],
                          signalName = "vbf_hh_l1cvv2cv1.root",
                          bkgName = "ttbar_PwPy8EG.root"):
  import ROOT

  # numerical library
  import numpy as np
  # to manipulate data using DataFrame
  import pandas as pd

  f = {}
  f["sig"] = ROOT.TFile(signalName)
  f["bkg"] = ROOT.TFile(bkgName)

  t = {}
  listBranches = [
   # used in the paper
   'diHiggsM', 'diTauMMCM', 'diJetM', 'diTauDR', 'diJetDR', 'METCentrality',
   #'diHiggsPt',
   # extra for VBF
   'diJetVBFM', 'diJetVBFPt',
   # other extra
   #'NJets', 'NJetsbtagged', 'diHiggsPt',
  # 'NJets', 'NJetsbtagged',
  # 'Tau1Pt', 'Tau1Eta', 'Tau1Phi',
  # 'Tau2Pt', 'Tau2Eta', 'Tau2Phi',
  # # uncomment to use MMC
  # #'diTauMMCM', 'diTauMMCPt', 'diTauMMCEta', #'diTauMMCPhi', 'diTauDR',
  # # to use only visible (for tests)
  # 'diTauVisM', 'diTauVisPt', 'diTauVisEta', 'diTauVisPhi', 'diTauVisDR',
  # 'Jet1Pt', 'Jet1Eta', 'Jet1Phi',
  # 'Jet2Pt', 'Jet2Eta', 'Jet2Phi',
  # 'diJetM', 'diJetPt', 'diJetEta', 'diJetPhi', 'diJetDR',
  # 'diHiggsM', 'diHiggsPt', 'diJetdiTauDR', #'MET',
  # # extra for VBF
  # 'diJetVBFM', 'diJetVBFPt', 'diJetVBFDR', 'diJetVBFDEta',
   ]

  hdf = pd.HDFStore('input.h5', 'w')
  Nrows = 0
  for treeName in treeNameList:
    t[treeName] = {}
    ptDrop = False
    for k in f:
      nn = treeName
      if "slope" in nn:
        nn = "Nominal"
        ptDrop = True
      t[treeName][k] = f[k].Get(nn)

    # sample specifies if it is signal or background
    Nrows += (t[treeName]["bkg"].GetEntries()+t[treeName]["sig"].GetEntries())

  df = pd.DataFrame(np.zeros((Nrows, len(listBranches)+3), dtype = np.float32), columns = ['sample', 'syst', 'weight']+listBranches)
  idx = 0
  for treeName in treeNameList:
    for sample in range(0, 2): # signal and bkg
      if sample == 0: sampleName = "bkg"
      else: sampleName = "sig"
      print("Reading sample %s" % sampleName)

      for k in range(0, t[treeName][sampleName].GetEntries()):
        t[treeName][sampleName].GetEntry(k)

        if k % 1000 == 0: print("Entry %d/%d" % (k, t[treeName][sampleName].GetEntries()))

        if t[treeName][sampleName].NJetsbtagged < 2: continue # added cut to follow paper

        if ptDrop and sample == 0:
          prob = 1.0
          if t[treeName][sampleName].diHiggsM < 400:
            prob = 0.95
          elif t[treeName][sampleName].diHiggsPt < 600:
            prob = 0.90
          elif t[treeName][sampleName].diHiggsPt < 800:
            prob = 0.85
          else:
            prob = 0.70
          r = np.random.uniform(0.0, 1.0)
          if r > prob:
            continue

        df.loc[idx, 'sample'] = sample # 0 for bkg, 1 for signal
        df.loc[idx, 'syst'] = float(treeName != 'Nominal') # 0 for nominal, 1 for syst
        df.loc[idx, 'weight'] = t[treeName][sampleName].EventWeight
        for br in range(0, len(listBranches)):
          df.loc[idx, listBranches[br]] = getattr(t[treeName][sampleName], listBranches[br])
        idx += 1
    if idx < Nrows:
      df = df[0:idx]
  hdf.put('df', df, format = 'table', data_columns = True)
  p = np.random.permutation(Nrows)
  ptrain = p[:int(Nrows/2)]
  ptest = p[int(Nrows/2):]
  train_bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index.isin(ptrain))))
  train_sig = pd.DataFrame( ((df['sample'] == 1) & (df.index.isin(ptrain))))
  train_syst = pd.DataFrame( ((df['syst'] == 1) & (df.index.isin(ptrain))))
  train_nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index.isin(ptrain))))
  test_bkg = pd.DataFrame( ((df['sample'] == 0) & (df.index.isin(ptest))))
  test_sig = pd.DataFrame( ((df['sample'] == 1) & (df.index.isin(ptest))))
  test_syst = pd.DataFrame( ((df['syst'] == 1) & (df.index.isin(ptest))))
  test_nominal = pd.DataFrame( ((df['syst'] == 0) & (df.index.isin(ptest))))
  hdf.put('train_bkg', train_bkg, format = 'table')
  hdf.put('train_sig', train_sig, format = 'table')
  hdf.put('train_syst', train_syst, format = 'table')
  hdf.put('train_nominal', train_nominal, format = 'table')
  hdf.put('test_bkg', test_bkg, format = 'table')
  hdf.put('test_sig', test_sig, format = 'table')
  hdf.put('test_syst', test_syst, format = 'table')
  hdf.put('test_nominal', test_nominal, format = 'table')
  hdf.close()

def plotRatio(num = "slope", den = "Nominal", var = "diHiggsPt"):
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import copy
  dfNum = pd.read_hdf('input_%s.h5' % num, 'df')
  dfDen = pd.read_hdf('input_%s.h5' % den, 'df')
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
  c, bins = np.histogram(dfNum[var], normed = False, bins = 20, weights = dfNum["EventWeight"])
  for k in range(len(bins-1)):
    if np.sum(c[0:k])/np.sum(c) > 0.99:
      bins = np.arange(bins[0], bins[k], (bins[k] - bins[0])/20)
      break
  c, b = np.histogram(dfNum[var], normed = False, bins = bins, weights = dfNum["EventWeight"])
  ns.append(c)
  c, b = np.histogram(dfDen[var], normed = False, bins = bins, weights = dfDen["EventWeight"])
  ns.append(c)
  c, b = np.histogram(dfNum[var], normed = False, bins = bins, weights = dfNum["EventWeight"]**2)
  ns_err.append(np.sqrt(c))
  c, b = np.histogram(dfDen[var], normed = False, bins = bins, weights = dfDen["EventWeight"]**2)
  ns_err.append(np.sqrt(c))
  binwidth = bins[1]-bins[0]
  s = binwidth*0.5
  
  ax1.step(bins[:-1], ns[0], alpha = 0.8, label = num, color = 'r')
  ax1.step(bins[:-1], ns[1], alpha = 0.8, label = den, color = 'b')

  ax1.errorbar(bins[:-2]+s, ns[0][1:], ns_err[0][1:], alpha = 0.8, color = 'r', fmt = 'o')
  ax1.errorbar(bins[:-2]+s, ns[1][1:], ns_err[1][1:], alpha = 0.8, color = 'b', fmt = 'o')

  ax1.set_ylim([0, None])
  ax1.legend()

  rat = copy.deepcopy(ns[0])
  rat_err = copy.deepcopy(ns_err[0])
  for k in range(0, len(rat)):
    rat[k] = 0
    rat_err[k] = 0
    if ns[1][k] != 0:
      rat[k] = ns[0][k]/ns[1][k]
      rat_err[k] = rat[k]*np.sqrt( (ns_err[0][k]/ns[0][k])**2 + (ns_err[1][k]/ns[1][k])**2 )
 
  ax2.step(bins[:-1], rat, alpha=0.8, color = 'k')
  ax2.errorbar(bins[:-2]+s, rat[1:], rat_err[1:], alpha=0.8, color = 'k', fmt = 'o')

  ax1.grid(linestyle = '-')
  ax2.grid(linestyle = '-')
  ax1.set_ylabel("Entries")
  ax2.set_ylabel('Ratio (%s/%s)' % (num, den))
  ax2.set_xlabel(var)
  ax2.set_ylim(0.5, 1.5)
  yticks = ax1.yaxis.get_major_ticks() 
  yticks[0].label1.set_visible(False)
  plt.savefig("rootToPandas_ratio_%s_%s_%s.pdf" % (var, num, den))
  plt.close("all")


# only needs to be called once: to make the data_Nominal.h5 file
# using the h5 file is faster than using ROOT files
transformROOTToPandas()

#for var in ["diHiggsM", "diJetM", "diTauMMCM"]:
#  plotRatio(var = var)
