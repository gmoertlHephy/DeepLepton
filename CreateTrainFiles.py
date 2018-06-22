import ROOT
import os
from random import randint

#merge infiles to TChain
ch=ROOT.TChain("tree")

#add local files to TChain
TChainFiles = open("TChainFiles.txt", "r")
FileList = TChainFiles.readlines()
for TChainFile in FileList:
    TChainFile = TChainFile.strip()
    ch.Add(TChainFile)    
    print(TChainFile+" added to TChain")

#add dpm files to TChain
TChainFiles = open("TChainFiles_crab.txt", "r")
FileList = TChainFiles.readlines()
for TChainFile in FileList:
    TChainFile = TChainFile.strip()
    ch.Add("root://hephyse.oeaw.ac.at//dpm/oeaw.ac.at/home/cms/store/user/gmortl/cmgTuples/lepton/"+TChainFile)    
    print(TChainFile+" added to TChain")

#get numbers of entries and minimum of Prompt/NonPrompt/Fake
n_lep = ch.GetEntries()
print (str(n_lep)+" entries in TChain")
n_ele = ch.GetEntries("abs(lep_pdgId)==11")
n_muo = ch.GetEntries("abs(lep_pdgId)==13")
n_ele_class = [ch.GetEntries("abs(lep_pdgId)==11 && lep_isPromptId==1"), ch.GetEntries("abs(lep_pdgId)==11 && lep_isNonPromptId==1"), ch.GetEntries("abs(lep_pdgId)==11 && lep_isFakeId==1")]
n_muo_class = [ch.GetEntries("abs(lep_pdgId)==13 && lep_isPromptId==1"), ch.GetEntries("abs(lep_pdgId)==13 && lep_isNonPromptId==1"), ch.GetEntries("abs(lep_pdgId)==13 && lep_isFakeId==1")]
n_ele_min = min(n_ele_class)
n_muo_min = min(n_muo_class)

print("Electrons: %i, Prompt: %i, NonPrompt: %i, Fake: %i in TChain" %(n_ele, n_ele_class[0], n_ele_class[1],n_ele_class[2]) if n_ele==sum(n_ele_class) else "Warning: wrong Prompt/NonPrompt/Fake Id entries")
print("Muons: %i, Prompt: %i, NonPrompt: %i, Fake: %i in TChain"     %(n_muo, n_muo_class[0], n_muo_class[1],n_muo_class[2]) if n_muo==sum(n_muo_class) else "Warning: wrong Prompt/NonPrompt/Fake Id entries")

#define list structure for loops
ele_par = [
            ['elePrompt.root',"abs(lep_pdgId)==11 && lep_isPromptId==1"],
            ['eleNonPrompt.root',"abs(lep_pdgId)==11 && lep_isNonPromptId==1"],
            ['eleFake.root',"abs(lep_pdgId)==11 && lep_isFakeId==1"]
          ]

muo_par = [
            ['muoPrompt.root',"abs(lep_pdgId)==13 && lep_isPromptId==1"],
            ['muoNonPrompt.root',"abs(lep_pdgId)==13 && lep_isNonPromptId==1"],
            ['muoFake.root',"abs(lep_pdgId)==13 && lep_isFakeId==1"]
          ]

lep_par = [ele_par, muo_par]

#fill separation files (Prompt, NonPrompt, Fake)
for lep in lep_par:
    for lep_class in lep: 
        sfile = ROOT.TFile(lep_class[0], 'recreate')
        sTree = ch.CopyTree(lep_class[1])
        sTree.AutoSave()
        print (str(ch.GetEntries(lep_class[1]))+" entries copied to "+lep_class[0] if ch.GetEntries(lep_class[1])==sTree.GetEntries() else "Warning: wrong number of entries copied")
        sfile.Close()

#read separation files
ele_files = []
for lep_class in ele_par:
    ele_files.append(ROOT.TFile(lep_class[0], 'read'))
muo_files = []
for lep_class in muo_par:
    muo_files.append(ROOT.TFile(lep_class[0], 'read'))
lep_files = [ele_files, muo_files]

#merge separation files to train file
for lep in lep_files:
    filename = ("trainfile_ele.root" if lep==ele_files else "trainfile_muo.root")
    trainfile = ROOT.TFile(filename, 'recreate')
    trainfile_tree = ch.CloneTree(0,"")
    
    nmin = n_ele_min if lep==ele_files else n_muo_min
    for i in xrange(nmin):
        for sfile in lep:
            sfile_tree = sfile.Get('tree')
            #n_sfile=sfile_tree.GetEntries()
            #j=(i if n_sfile==nmin else randint(0,n_sfile-1)) #randomize index if sfile has more entries than nmin
            #print (i,j)
            a=sfile_tree.GetEntry(i)                          
            sfile_tree.CopyAddresses(trainfile_tree)
            b=trainfile_tree.Fill()
            if a!=b: print ("error when copying entry "+i)
    
    for sfile in lep:
        sfile.Close()
    
    trainfile_tree.AutoSave()
    print ("3 x %i = %i entries copied to %s" % (nmin, trainfile_tree.GetEntries(), filename))
    trainfile.Close()
    
    
