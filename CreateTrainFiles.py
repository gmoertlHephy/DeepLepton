# Standard imports
import ROOT
import os

from fnmatch import fnmatch
import numpy.random
# RootTools
from RootTools.core.Sample import *

####################################
####### splittedLeptonFiles ########
####################################
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


####################################
######## splittedTrainFiles ########
####################################

#read class files
ele_files = []
for lep_class in ele_par:
    ele_files.append(ROOT.TFile(lep_class["filename"], 'read'))
muo_files = []
for lep_class in muo_par:
    muo_files.append(ROOT.TFile(lep_class["filename"], 'read'))
lep_files = [ele_files, muo_files]

n_ele_prompt = ele_files[0].Get('tree').GetEntries()
n_ele_nonpormpt = ele_files[1].Get('tree').GetEntries()
n_ele_fake = ele_files[2].Get('tree').GetEntries()
n_ele = [n_ele_prompt, n_ele_nonpormpt, n_ele_fake]

n_muo_prompt = muo_files[0].Get('tree').GetEntries()
n_muo_nonpormpt = muo_files[1].Get('tree').GetEntries()
n_muo_fake = muo_files[2].Get('tree').GetEntries()
n_muo = [n_muo_prompt, n_muo_nonpormpt, n_muo_fake]

n_ele_min=min(n_ele)
n_muo_min=min(n_muo)

#print n_ele_min, n_muo_min

ch = ROOT.TChain('tree')
ch.Add('classsamples/muoFake.root')

#merge separation files to train file
for lep in lep_files:

    nmin = n_ele_min if lep==ele_files else n_muo_min
    n_maxfileentries=3*1000000
    n_actualentries=0
    n_file=1
    fileprefix = ("trainsamples/trainfile_ele_" if lep==ele_files else "trainsamples/trainfile_muo_")

    for i in xrange(nmin):

        if n_actualentries==0 and n_file==1:
            trainfile = ROOT.TFile(fileprefix+str(n_file)+".root", 'recreate')
            trainfile_tree = ch.CloneTree(0,"")
        if n_actualentries==0 and n_file>=2:
            print ("%i entries copied to %s" % (trainfile_tree.GetEntries(), fileprefix+str(n_file-1)+".root"))
            trainfile.Write(fileprefix+str(n_file-1)+".root", trainfile.kOverwrite)
            trainfile.Close()
            trainfile = ROOT.TFile(fileprefix+str(n_file)+".root", 'recreate')
            trainfile_tree = ch.CloneTree(0,"")

        for sfile in lep:
            sfile_tree = sfile.Get('tree')
            n_entries = sfile_tree.GetEntries()
            j = int(n_entries/nmin*i)
            #j = numpy.random.randint(0,n_entries)
            #print (i, j)
            a=sfile_tree.GetEntry(j)
            sfile_tree.CopyAddresses(trainfile_tree)
            b=trainfile_tree.Fill()
            if a!=b: print ("error when copying entry "+i)

        n_actualentries += 3
        if n_actualentries>=n_maxfileentries:
            n_actualentries=0
            n_file += 1

    print ("%i entries copied to %s" % (trainfile_tree.GetEntries(), fileprefix+str(n_file)+".root"))
    trainfile.Write(fileprefix+str(n_file)+".root", trainfile.kOverwrite)
    trainfile.Close()

    for sfile in lep:
        sfile.Close()
