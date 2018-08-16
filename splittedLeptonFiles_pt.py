# Standard imports
import ROOT
import os

from fnmatch import fnmatch
#from def_preprocessing import getPtCuts

## RootTools
#from RootTools.core.Sample import *

#make FileList
pathname  = '/afs/hephy.at/data/rschoefbeck02/cmgTuples/lepton/'
pattern   = 'tree.root'
writePath = '/afs/hephy.at/work/g/gmoertl/CMSSW_9_4_6_patch1/src/DeepLepton/preprocessing/classsamples_pt/'


FileList=[]
for path, subdirs, files in os.walk(pathname):
    for name in files:
        if fnmatch(name, pattern):
            FileList.append(os.path.join(path, name))
            #print(os.path.join(path, name)
print "%i files found" %(len(FileList))

#define class file parameters
ele_par = [
            {"filename": 'elePrompt',"lep_pdgId": 11, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'eleNonPrompt',"lep_pdgId": 11, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'eleFake',"lep_pdgId": 11, "lep_isClassId": 'lep_isFakeId'},
          ]

muo_par = [
            {"filename": 'muoPrompt',"lep_pdgId": 13, "lep_isClassId": 'lep_isPromptId'},
            {"filename": 'muoNonPrompt',"lep_pdgId": 13, "lep_isClassId": 'lep_isNonPromptId'},
            {"filename": 'muoFake',"lep_pdgId": 13, "lep_isClassId": 'lep_isFakeId'},
          ]

lep_par = [ele_par, muo_par]

#pt_cuts=getPtCuts()
pt_cuts = [
           [0,5], [5,10],
           [10,15], [15,20], [20,25],
           [25,30], [30,40], [40,55], [55,75], [75,100], [100,150], [150,250], [250,400], [400,600] 
          ]

#TChain for CloneTree to class files
ch = ROOT.TChain('tree')
ch.Add(FileList[0])

#TestFileList=[FileList[0], FileList[1200], FileList[2400]]
#FileList = TestFileList

#for chainFile in FileList:
#    ch.Add(chainFile)

#for lep in lep_par:
#    for lep_class in lep: 
#        for pt_cut in pt_cuts:
#
#            selectionString = 'abs(lep_pdgId)=='+str(lep_class["lep_pdgId"])+'&&'+lep_class["lep_isClassId"]+'==1&&lep_pt>'+str(pt_cut[0])+'&&lep_pt<='+str(pt_cut[1])
#            #chainTree = ch.GetTree()
#            fileName = lep_class["filename"]+'_pt'+str(pt_cut[0])+'to'+str(pt_cut[1])+'.root'
#            classfile = ROOT.TFile(fileName, 'recreate')
#            classfileTree = ch.CopyTree(selectionString)
#
#            classfile.Write(fileName, classfile.kOverwrite)
#            classfile.Close()
#            print fileName+" (re)created"

#fill class files (Prompt, NonPrompt, Fake)
for lep in lep_par:
    for lep_class in lep:
        for pt_cut in pt_cuts:

            fileName = lep_class["filename"]+'_pt'+str(pt_cut[0])+'to'+str(pt_cut[1])+'.root'
            classfile = ROOT.TFile(writePath+fileName, 'recreate')
            classfileTree = ch.CloneTree(0,"")
            for rootfile in FileList:
                datafile = ROOT.TFile(rootfile, 'read')
                datafileTree = datafile.Get('tree')
                datafileTree.CopyAddresses(classfileTree)
                
                pdgId     = datafileTree.GetLeaf("lep_pdgId")
                isClassId = datafileTree.GetLeaf(lep_class["lep_isClassId"])
                pt        = datafileTree.GetLeaf("lep_pt")
                
                for i in xrange(datafileTree.GetEntries()):
                    pdgId.GetBranch().GetEntry(i)
                    isClassId.GetBranch().GetEntry(i)
                    pt.GetBranch().GetEntry(i)
                    lep_pdgId     = pdgId.GetValue()
                    lep_isClassId = isClassId.GetValue()
                    lep_pt        = pt.GetValue()

                    #print lep_pdgId, lep_isClassId 
                    if abs(lep_pdgId)==lep_class["lep_pdgId"] and lep_isClassId==1 and lep_pt>pt_cut[0] and lep_pt<=pt_cut[1]:
                        a = datafileTree.GetEntry(i) 
                        b = classfileTree.Fill()
            
            print ("%i entries copied to %s" %(classfileTree.GetEntries(), fileName))
            classfile.Write(writePath+fileName, classfile.kOverwrite)
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
