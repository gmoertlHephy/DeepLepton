# Standard imports
import ROOT
import os

#define list structure for loops
ele_par = [
            {"filename": 'classsamples/elePrompt.root', "shortname": "elePrompt", "lep_pdgId": 11, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'classsamples/eleNonPrompt.root', "shortname": "eleNonPrompt", "lep_pdgId": 11, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'classsamples/eleFake.root', "shortname": "eleFake", "lep_pdgId": 11, "lep_isClassId": 'lep_isFakeId'},
          ]

muo_par = [
            {"filename": 'classsamples/muoPrompt.root', "shortname": "muoPrompt", "lep_pdgId": 13, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'classsamples/muoNonPrompt.root', "shortname": "muoNonPrompt", "lep_pdgId": 13, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'classsamples/muoFake.root', "shortname": "muoFake", "lep_pdgId": 13, "lep_isClassId": 'lep_isFakeId'},
          ]

lep_par = [ele_par, muo_par]

#define cuts for electrons and muons (Working points for 2016 data for 80X)
eleCuts_tight="(abs(lep_etaSc)<=1.479&&lep_full5x5_sigmaIetaIeta<0.00998&&abs(lep_dEtaInSeed)<0.00308&&abs(lep_dPhiScTrkIn)<0.0816&&lep_hadronicOverEm<0.0414&&abs(lep_eInvMinusPInv)<0.0129&&lep_lostOuterHits<=1&&lep_convVeto==1)||(abs(lep_etaSc)> 1.479&&lep_full5x5_sigmaIetaIeta<0.0292 &&abs(lep_dEtaInSeed)<0.00605&&abs(lep_dPhiScTrkIn)<0.0394&&lep_hadronicOverEm<0.0641&&abs(lep_eInvMinusPInv)<0.0129&&lep_lostOuterHits<=1&&lep_convVeto==1)"
muoCuts_tight="lep_isGlobalMuon==1&&lep_pfMuonId==1&&lep_globalTrackChi2<10.&&lep_trackerHits>0&&lep_nStations>1&&abs(lep_dxy)<0.2&&abs(lep_dz)<0.5&&lep_pixelLayers>0&&lep_trackerLayers>5"

#read class files
ele_files = []
for lep_class in ele_par:
    ele_files.append([ROOT.TFile(lep_class["filename"], 'read'), lep_class["shortname"]])
muo_files = []
for lep_class in muo_par:
    muo_files.append([ROOT.TFile(lep_class["filename"], 'read'), lep_class["shortname"]])
lep_files = [ele_files, muo_files]


for lep in lep_files:
    for sfile in lep:
        print sfile
        sfile_tree = sfile[0].Get('tree')
        selectionString = eleCuts_tight if lep == ele_files else muoCuts_tight
        nfile = ROOT.TFile('classsamples_isolation/'+sfile[1]+'_tight.root', 'recreate')
        nfile_tree = sfile_tree.CopyTree(selectionString)
        nfile.Write()
        nfile.Close()
        #print sfile_tree.GetEntries()
        sfile[0].Close()

