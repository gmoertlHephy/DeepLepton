# Standard imports
import ROOT
import os

from fnmatch import fnmatch

# RootTools
from RootTools.core.Sample import *

#make FileList
pathname = '/afs/hephy.at/data/rschoefbeck02/cmgTuples/lepton/'
pattern  = 'tree.root'

FileList=[]
for path, subdirs, files in os.walk(pathname):
    for name in files:
        if fnmatch(name, pattern):
            FileList.append(os.path.join(path, name))
            #print(os.path.join(path, name)
print "%i files found" %(len(FileList))

#define class file parameters
ele_par = [
            {"filename": 'classsamples/elePrompt.root',"lep_pdgId": 11, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'classsamples/eleNonPrompt.root',"lep_pdgId": 11, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'classsamples/eleFake.root',"lep_pdgId": 11, "lep_isClassId": 'lep_isFakeId'},
          ]

muo_par = [
            {"filename": 'classsamples/muoPrompt.root',"lep_pdgId": 13, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'classsamples/muoNonPrompt.root',"lep_pdgId": 13, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'classsamples/muoFake.root',"lep_pdgId": 13, "lep_isClassId": 'lep_isFakeId'},
          ]

lep_par = [ele_par, muo_par]

#TChain for CloneTree to class files
TestFileList=[FileList[0], FileList[1200], FileList[2400]]
ch = ROOT.TChain('tree')
ch.Add(TestFileList[0])

#FileList = TestFileList

#fill class files (Prompt, NonPrompt, Fake)
for lep in lep_par:
    for lep_class in lep: 
        classfile = ROOT.TFile(lep_class["filename"], 'recreate')
        classfileTree = ch.CloneTree(0,"")
        for rootfile in FileList:
            datafile = ROOT.TFile(rootfile, 'read')
            datafileTree = datafile.Get('tree')
            datafileTree.CopyAddresses(classfileTree)
            
            pdgId     = datafileTree.GetLeaf("lep_pdgId")
            isClassId = datafileTree.GetLeaf(lep_class["lep_isClassId"])
            
            for i in xrange(datafileTree.GetEntries()):
                pdgId.GetBranch().GetEntry(i)
                isClassId.GetBranch().GetEntry(i)    
                lep_pdgId     = pdgId.GetValue()
                lep_isClassId = isClassId.GetValue()

                #print lep_pdgId, lep_isClassId 
                if abs(lep_pdgId)==lep_class["lep_pdgId"] and lep_isClassId==1:
                    a = datafileTree.GetEntry(i) 
                    b = classfileTree.Fill()
        
        print ("%i entries copied to %s" %(classfileTree.GetEntries(), lep_class["filename"]))
        classfile.Write(lep_class["filename"], classfile.kOverwrite)
        classfile.Close()

###########################
######## old stuff ########
###########################
#
#subdirs = os.listdir(path)
#print(subdirs)
#CrabSamples=[]
#for subdir in subdirs:
#    CrabSamples.append(Sample.fromCMGOutput(name="", baseDirectory=path+subdir, treeFilename = 'tree.root', chunkString = None, treeName = 'tree', maxN = None, 
#                                            selectionString = None, weightString = None, isData = False, color = 0, texName = None))
#CrabSample=Sample.combine(name="CrabSample", samples=CrabSamples, texName = None, maxN = None, color = 0)
#CrabSampleTreeName=CrabSample.treeName
#
##get numbers of entries and minimum of Prompt/NonPrompt/Fake
#n_lep = ch.GetEntries()
#print (str(n_lep)+" entries in TChain")
#n_ele = ch.GetEntries("abs(lep_pdgId)==11")
#n_muo = ch.GetEntries("abs(lep_pdgId)==13")
#n_ele_class = [ch.GetEntries("abs(lep_pdgId)==11 && lep_isPromptId==1"), ch.GetEntries("abs(lep_pdgId)==11 && lep_isNonPromptId==1"), ch.GetEntries("abs(lep_pdgId)==11 && lep_isFakeId==1")]
#n_muo_class = [ch.GetEntries("abs(lep_pdgId)==13 && lep_isPromptId==1"), ch.GetEntries("abs(lep_pdgId)==13 && lep_isNonPromptId==1"), ch.GetEntries("abs(lep_pdgId)==13 && lep_isFakeId==1")]
#n_ele_min = min(n_ele_class)
#n_muo_min = min(n_muo_class)
#
#print("Electrons: %i, Prompt: %i, NonPrompt: %i, Fake: %i in TChain" %(n_ele, n_ele_class[0], n_ele_class[1],n_ele_class[2]) if n_ele==sum(n_ele_class) else "Warning: wrong Prompt/NonPrompt/Fake Id entries")
#print("Muons: %i, Prompt: %i, NonPrompt: %i, Fake: %i in TChain"     %(n_muo, n_muo_class[0], n_muo_class[1],n_muo_class[2]) if n_muo==sum(n_muo_class) else "Warning: wrong Prompt/NonPrompt/Fake Id entries")
