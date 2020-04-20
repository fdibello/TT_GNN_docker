import ROOT
import random
from array import array
import math
import  numpy as np
def gau(n,m,s):
 Gaus_pt =  ROOT.TF1("ms","gaus(0)",-1000,1000)
 Gaus_pt.SetParameter(0,n)
 Gaus_pt.SetParameter(1,m)
 Gaus_pt.SetParameter(2,s)
 return Gaus_pt

def DefineKine(eta,phi,dR):
  etaJ = 10000
  phiJ = 10000
  while(abs(etaJ)>2.4 or abs(phiJ)>3.13 ):
   theta = random.uniform(-3.14, 3.14)
   sign = 1
   signP = 1
   if eta > 0: sign = -1
   if phi > 0: signP = -1
  # print eta, etaJ
   etaJ =  eta + sign* dR* math.cos(theta)
   phiJ =  phi + signP*dR* math.sin(theta)
  return etaJ, phiJ

def ComputeEffDR(jet,alljet,efB,efC,efL):
  eff = 1
  ef = 1
  for j in alljet:
    if jet == j: continue
    if j[0] == -1: continue
    deta = jet[1]-j[1]
    dphi = jet[2]-j[2]
    dR = np.sqrt(deta*deta+dphi*dphi)
    if j[5] == 1: ef = efB.GetBinContent(efB.GetXaxis().FindBin(dR))
    if j[5] == 2: ef = efC.GetBinContent(efB.GetXaxis().FindBin(dR))
    if j[5] == 3: ef = efL.GetBinContent(efB.GetXaxis().FindBin(dR))
    if ef > 1: ef = 1
    if ef != 0: eff = ef*eff
#    if jet == alljet[1]: print dR, ef, "jet number 2", eff
#    if jet == alljet[0]: print dR, ef, "jet number 1", eff

  return eff

jet_pt = ROOT.vector('float')()
jet_eta = ROOT.vector('float')()
jet_phi = ROOT.vector('float')()
jet_label = ROOT.vector('int')()
jet_eff = ROOT.vector('float')()
jet_score = ROOT.vector('float')()
jet_tag = ROOT.vector('int')()
t_nJets     = array( 'i', [ 0 ] )
t = ROOT.TTree( 't1', 'tree with histos' )
#t_pT      = array( 'f', [ 0 ] )
#t_pT2      = array( 'f', [ 0 ] )
t_mass    = array( 'f', [ 0 ] )
#t_eta     = array( 'f', [ 0 ] )
#t_phi     = array( 'f', [ 0 ] )
#t_eta2     = array( 'f', [ 0 ] )
#t_phi2     = array( 'f', [ 0 ] )
#t_dR      = array( 'f', [ 0 ] )
#t_label   = array( 'i', [ 0 ] )
#t_tag     = array( 'i', [ 0 ] )
#t_dR2      = array( 'f', [ 0 ] )
#t_tag2     = array( 'i', [ 0 ] )

#t_dR_2      = array( 'f', [ 0 ] )
#t_dR_3      = array( 'f', [ 0 ] )
#t_dR_4      = array( 'f', [ 0 ] )
#t_dR_5      = array( 'f', [ 0 ] )
#t_dR2_2      = array( 'f', [ 0 ] )
#t_dR2_3      = array( 'f', [ 0 ] )
#t_dR2_4      = array( 'f', [ 0 ] )
#t_dR2_5      = array( 'f', [ 0 ] )
#t_label2   = array( 'i', [ 0 ] )
#t_label2_2   = array( 'i', [ 0 ] )
#t_label2_3   = array( 'i', [ 0 ] )
#t_label2_4   = array( 'i', [ 0 ] )
#t_label2_5   = array( 'i', [ 0 ] )
#t_label_2   = array( 'i', [ 0 ] )
#t_label_3   = array( 'i', [ 0 ] )
#t_label_4   = array( 'i', [ 0 ] )
#t_label_5   = array( 'i', [ 0 ] )

# vectors
t.Branch( 'jet_pt', jet_pt )
t.Branch( 'jet_eta', jet_eta )
t.Branch( 'jet_phi', jet_phi )
t.Branch( 'jet_label', jet_label )
t.Branch( 'jet_score', jet_score )
t.Branch( 'jet_eff', jet_eff )
t.Branch( 'jet_tag', jet_tag )
t.Branch( 'nJets', t_nJets, 'nJets/I' )

#t.Branch( 'pT', t_pT, 'pT/F' )
t.Branch( 'mass', t_mass, 'mass/F' )
#t.Branch( 'eta', t_eta, 'eta/F' )
#t.Branch( 'phi', t_phi, 'eta/F' )
#
#t.Branch( 'dR1', t_dR, 'dR1/F' )
#
#t.Branch( 'label', t_label, 'label/I' )
#t.Branch( 'isTag', t_tag, 'istag/I' )
#t.Branch( 'pT2', t_pT2, 'pT2/F' )
#t.Branch( 'eta2', t_eta2, 'eta2/F' )
#t.Branch( 'phi2', t_phi2, 'phi2/F' )
#t.Branch( 'isTag2', t_tag2, 'istag2/I' )

#t_pT3       = array( 'f', [ 0 ] )
#t_pT4       = array( 'f', [ 0 ] )
#t_pT5       = array( 'f', [ 0 ] )
#t_pT6       = array( 'f', [ 0 ] )
#t_eta3      = array( 'f', [ 0 ] )
#t_eta4      = array( 'f', [ 0 ] )
#t_eta5      = array( 'f', [ 0 ] )
#t_eta6      = array( 'f', [ 0 ] )
#t_phi3      = array( 'f', [ 0 ] )
#t_phi4      = array( 'f', [ 0 ] )
#t_phi5      = array( 'f', [ 0 ] )
#t_phi6      = array( 'f', [ 0 ] )
#t_isTag3    = array( 'i', [ 0 ] )
#t_isTag4    = array( 'i', [ 0 ] )
#t_isTag5    = array( 'i', [ 0 ] )
#t_isTag6    = array( 'i', [ 0 ] )
#t_label3    = array( 'i', [ 0 ] )
#t_label4    = array( 'i', [ 0 ] )
#t_label5    = array( 'i', [ 0 ] )
#t_label6    = array( 'i', [ 0 ] )
#t_label6    = array( 'i', [ 0 ] )
#t_label1    = array( 'i', [ 0 ] )
#t_label2    = array( 'i', [ 0 ] )

#t.Branch( 'pT3', t_pT3, 'pT3/F' )
#t.Branch( 'pT4', t_pT4, 'pT4/F' )
#t.Branch( 'pT5', t_pT5, 'pT5/F' )
#t.Branch( 'pT6', t_pT6, 'pT6/F' )
#
#t.Branch( 'eta3', t_eta3, 'eta3/F' )
#t.Branch( 'eta4', t_eta4, 'eta4/F' )
#t.Branch( 'eta5', t_eta5, 'eta5/F' )
#t.Branch( 'eta6', t_eta6, 'eta6/F' )
#
#t.Branch( 'phi3', t_phi3, 'phi3/F' )
#t.Branch( 'phi4', t_phi4, 'phi4/F' )
#t.Branch( 'phi5', t_phi5, 'phi5/F' )
#t.Branch( 'phi6', t_phi6, 'phi6/F' )
#
#t.Branch( 'label1', t_label1, 'label1/I' )
#t.Branch( 'label2', t_label2, 'label2/I' )
#t.Branch( 'label3', t_label3, 'label3/I' )
#t.Branch( 'label4', t_label4, 'label4/I' )
#t.Branch( 'label5', t_label5, 'label5/I' )
#t.Branch( 'label6', t_label6, 'label6/I' )
#
#t.Branch( 'isTag3', t_isTag3, 'isTag3/I' )
#t.Branch( 'isTag4', t_isTag4, 'isTag4/I' )
#t.Branch( 'isTag5', t_isTag5, 'isTag5/I' )
#t.Branch( 'isTag6', t_isTag6, 'isTag6/I' )

f1 = ROOT.TFile.Open("../efficiency.root")

efB=f1.Get('h_eff0')
efC=f1.Get('h_eff1')
efL=f1.Get('h_eff2')
efBpT=f1.Get('h_effpt0')
efCpT=f1.Get('h_effpt1')
efLpT=f1.Get('h_effpt2')
dRefB=f1.Get('h_dR0')
dRefC=f1.Get('h_dR1')
dRefL=f1.Get('h_dR2')

#now need to generate events according to some given distributions

nm = 1
sig = 200
G_pt = gau(nm,20,sig)
sig = 0.5
G_eta = gau(nm,0,sig)
sig = 100
G_phi = gau(nm,0,sig)
Gaus_dR =  ROOT.TF1("dr","sqrt(x)",0.011,100)

h = ROOT.TH1F( 'h1', 'testEff', 100, 20, 1000. )
hden = ROOT.TH1F( 'h1den', 'testEffden', 100, 20, 1000. )
he = ROOT.TH1F( 'h1eta', 'testEffeta', 30, -2., 2. )
heden = ROOT.TH1F( 'h1dene', 'testEffdene', 30, -2., 2. )
hdR = []
hdRden = []
for flav in range(3):
 hd = ROOT.TH1F( 'h1dR'+str(flav), 'testEffdR'+str(flav), 30, 0., 2. )
 hdendr = ROOT.TH1F( 'h1dendR'+str(flav), 'testEffdendR'+str(flav), 30, 0., 2. )
 hdR.append(hd)
 hdRden.append(hdendr)

f = ROOT.TFile( 'datasetB.root', 'recreate' )
Nevents = 10000
for n in range(Nevents):

 Clos1 = []
 Clos2 = []

 pT2 = G_pt.GetRandom(20,1000)
 pT = G_pt.GetRandom(20,1000)
 eta = G_eta.GetRandom(-2.,2.)
 phi = G_phi.GetRandom(-3.14,1.14)
 eta2 = eta + random.gauss(0.3,0.5)
 phi2 = phi + random.gauss(0,0.4)

 #Shall generalize to different
 dR = Gaus_dR.GetRandom(0.011,3)

 closestJets = []
 #filling here the nominal
 labelClosest = G_pt.GetRandom(1,4)
## labelClosest = 3
 closestJets.append([pT,eta,phi,2,dR,int(labelClosest)])
 #let's imagine we have the 2 jets + other 4. So we fix only the leading and
 #compute here the others
 #[pt,eta,phi,Mass,dR]
 Number_jets = 0
 JetA =  ROOT.TF1("jets","exp(-1.*x/4.)",0,5)
 Number_jets = int(JetA.GetRandom(0,5))

 for jets in range(1+Number_jets):
  jetScore = random.uniform(0,1)
#  always keep the subleading
  JdR = Gaus_dR.GetRandom(0.011,3)
  Jeta, Jphi = DefineKine(eta,phi,JdR)
  Jpt = 100000000
  while Jpt > pT or Jpt > pT2:
   Jpt = G_pt.GetRandom(20,1000)

  labelC = int(G_pt.GetRandom(1,4))
 ## labelC = 3
  JM = 2
  if jets == 0: closestJets.append([pT2,Jeta,Jphi,JM,JdR,labelC])
  else: closestJets.append([Jpt,Jeta,Jphi,JM,JdR,labelC])

### changed
 eta2 = closestJets[1][1]
 phi2 = closestJets[1][2]

 isTag = 0
 isTag1 = 0
 l1 = ROOT.TLorentzVector()
 l2 = ROOT.TLorentzVector()
 totl = ROOT.TLorentzVector()
 l1.SetPtEtaPhiM(pT,eta,phi,2)
 l2.SetPtEtaPhiM(pT2,eta2,phi2,2)
 totl.SetPtEtaPhiE(0,0,0,0)
 totl = (l1+l2)
 mass = totl.Mag()
 lab  = closestJets[0][5]
 lab1 = closestJets[1][5]

# if lab == 1: eff = efBpT.GetBinContent(efBpT.GetXaxis().FindBin(pT),efBpT.GetYaxis().FindBin(eta))
# if lab == 2: eff = efCpT.GetBinContent(efBpT.GetXaxis().FindBin(pT),efBpT.GetYaxis().FindBin(eta))
# if lab == 3: eff = efLpT.GetBinContent(efBpT.GetXaxis().FindBin(pT),efBpT.GetYaxis().FindBin(eta))
# if closestJets[1][5]  == 1: eff1 =  efBpT.GetBinContent(efBpT.GetXaxis().FindBin(pT2),efBpT.GetYaxis().FindBin(eta2))
# if closestJets[1][5]  == 2: eff1 =  efCpT.GetBinContent(efBpT.GetXaxis().FindBin(pT2),efBpT.GetYaxis().FindBin(eta2))
# if closestJets[1][5]  == 3: eff1 =  efLpT.GetBinContent(efBpT.GetXaxis().FindBin(pT2),efBpT.GetYaxis().FindBin(eta2))
 #correct for new efficiencies
 eff_clos = []
 sj = 0
 ef = 1
 efpT = 1
 closjet = []
 for jet in closestJets:
  sj = sj + 1
#  there is a bug  here, because the deltaR of the leading and subleading has
#  been computed twice
  ef = ComputeEffDR(jet,closestJets,dRefB,dRefC,dRefL)
  #if sj == 1: ef = ef*eff
  #elif sj == 2:
  #  ef = ef*eff1
  #else:
  if jet[5] == 1: efpT = efBpT.GetBinContent(efBpT.GetXaxis().FindBin(jet[0]),efBpT.GetYaxis().FindBin(jet[1]))
  if jet[5] == 2: efpT = efCpT.GetBinContent(efBpT.GetXaxis().FindBin(jet[0]),efBpT.GetYaxis().FindBin(jet[1]))
  if jet[5] == 3: efpT = efLpT.GetBinContent(efBpT.GetXaxis().FindBin(jet[0]),efBpT.GetYaxis().FindBin(jet[1]))
  ef = efpT * ef
  tg = 0
  rnd = random.uniform(0,1)
#  print ef, sj
  if ef > rnd: tg = 1
  #print "1-------- ",jet[5], ef
  eff_clos.append([ef,tg,rnd])
 #print " light ",ef, jet[5]
# if Number_jets == 1 : print "eff leading = ", eff_clos[0][0], eff_clos[1][0], eff1, ef, closestJets
 isTag  = eff_clos[0][1]
 isTag1 = eff_clos[1][1]

# now shall start filling up the trees
# t_pT[0] = pT
 t_mass[0] = mass
# t_eta[0] = eta
# t_phi[0] = phi
# t_dR[0] = dR
# t_label[0] = int(labelClosest)
# t_tag[0] = isTag
# t_pT2[0] = pT2
# t_eta2[0] = eta2
# t_phi2[0] = phi2
# t_tag2[0] = isTag1

# t_pT3[0]  = closestJets[2][0]
# t_eta3[0] = closestJets[2][1]
# t_phi3[0] = closestJets[2][2]
# t_isTag3[0] = eff_clos[2][1]
#
# t_pT4[0]  = closestJets[3][0]
# t_eta4[0] = closestJets[3][1]
# t_phi4[0] = closestJets[3][2]
# t_isTag4[0] = eff_clos[3][1]
#
# t_pT5[0]  = closestJets[4][0]
# t_eta5[0] = closestJets[4][1]
# t_phi5[0] = closestJets[4][2]
# t_isTag5[0] = eff_clos[4][1]
#
# t_pT6[0]  = closestJets[5][0]
# t_eta6[0] = closestJets[5][1]
# t_phi6[0] = closestJets[5][2]
# t_isTag6[0] = eff_clos[5][1]
#
# t_label1[0] = closestJets[0][5]
# t_label2[0] = closestJets[1][5]
# t_label3[0] = closestJets[2][5]
# t_label4[0] = closestJets[3][5]
# t_label5[0] = closestJets[4][5]
# t_label6[0] = closestJets[5][5]

 #order from the leading pT
 cJ = []
 j1 = -1
 for jet in closestJets:
  j1 = j1 + 1
  cJ.append(jet+eff_clos[j1])

 x = closestJets
 cJ.sort(reverse = True,key=lambda x: x[0])
 j = -1
 for je in cJ:
  j = j + 1
  # pt,eta, phi,  dR, label,mass, eff,score
  jet_pt.push_back(   je[0])
  jet_eta.push_back(  je[1])
  jet_phi.push_back(  je[2])
  jet_label.push_back(je[5])
  jet_eff.push_back(  je[6])
  jet_score.push_back(je[8])
  jet_tag.push_back(je[7])
#  print je[5],je[6]

 t_nJets[0] = j+1
# print closestJets
 t.Fill()
 jet_pt.clear()
 jet_eta.clear()
 jet_phi.clear()
 jet_label.clear()
 jet_eff.clear()
 jet_score.clear()
 jet_tag.clear()

 if isTag == 1:
     h.Fill(pT)
     hdR[lab-1].Fill(dR)
     he.Fill(eta)
 hden.Fill(pT)
 heden.Fill(eta)
 hdRden[lab-1].Fill(dR)

f.cd()
h.Divide(hden)
for ft in range(3):
   hdR[ft].Divide(hdRden[ft])
   hdR[ft].Write()
he.Divide(heden)
he.Write()
h.Write()
G_pt.Draw("same")
t.Write()
f.Close()
