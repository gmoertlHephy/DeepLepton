# Standard imports
import ROOT
import os

import numpy.random

from def_preprocessing import getPtCuts

#define list structure for loops
readPath = '/afs/hephy.at/work/g/gmoertl/CMSSW_9_4_6_patch1/src/DeepLepton/preprocessing/classsamples_pt/'
readFileSuffix = '.root'
writePath = '/afs/hephy.at/work/g/gmoertl/CMSSW_9_4_6_patch1/src/DeepLepton/preprocessing/trainsamples_pt_a90k/'
wirteFileSuffix = readFileSuffix

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

pt_cuts = getPtCuts()

for pt_cut in pt_cuts:

    ptSuffix = '_pt'+str(pt_cut[0])+'to'+str(pt_cut[1])
    
    #read class files
    ele_files = []
    for lep_class in ele_par:
        fileName = readPath+lep_class["filename"]+ptSuffix+readFileSuffix
        ele_files.append(ROOT.TFile(fileName, 'read'))
    muo_files = []
    for lep_class in muo_par:
        fileName = readPath+lep_class["filename"]+ptSuffix+readFileSuffix
        muo_files.append(ROOT.TFile(fileName, 'read'))
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
    ch.Add(fileName) #just add one file to chain for ch.CloneTree()

    #merge separation files to train file
    for lep in lep_files:

        nmin = n_ele_min if lep==ele_files else n_muo_min
        n_maxfileentries=3*30000
        n_actualentries=0
        n_file=1
        fileprefix = (writePath+'trainfile_ele'+ptSuffix+'_' if lep==ele_files else writePath+'trainfile_muo'+ptSuffix+'_')    
        
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
