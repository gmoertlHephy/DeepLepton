#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.models import Model
import warnings
warnings.warn("DeepJet_models.py is deprecated and will be removed! Please move to the models directory", DeprecationWarning)

from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D
#fix for dropout on gpus

#import tensorflow
#from tensorflow.python.ops import control_flow_ops 
#tensorflow.python.control_flow_ops = control_flow_ops

from TrainDataDeepLepton import TrainData_fullTruth
from TrainDataDeepLepton import fileTimeOut

class TrainData_deepLeptons_Leptons(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches(['lep_pt', 'lep_eta', 'lep_rho',
        'lep_etaSc', 'lep_full5x5_sigmaIetaIeta', 'lep_dEtaInSeed',
        'lep_dPhiScTrkIn', 'lep_relIso03', 'lep_eInvMinusPInv',
        'lep_lostHits_float', 'lep_convVeto_float', 'lep_hadronicOverEm', 
        'lep_isElectron_float', 'lep_isMuon_float',
        'lep_npfCands_neutral_float', 'lep_npfCands_charged_float', 'lep_npfCands_photon_float', 'lep_npfCands_electron_float', 'lep_npfCands_muon_float'
        ])

        self.addBranches(['pfCand_neutral_pt', 'pfCand_neutral_eta', 'pfCand_neutral_phi', 'pfCand_neutral_mass', 'pfCand_neutral_puppiWeight', 'pfCand_neutral_hcalFraction', 'pfCand_neutral_fromPV'],25)
        self.addBranches(['pfCand_charged_pt', 'pfCand_charged_eta', 'pfCand_charged_phi', 'pfCand_charged_mass', 'pfCand_charged_puppiWeight', 'pfCand_charged_hcalFraction', 'pfCand_charged_fromPV'],25)
        self.addBranches(['pfCand_photon_pt', 'pfCand_photon_eta', 'pfCand_photon_phi', 'pfCand_photon_mass', 'pfCand_photon_puppiWeight', 'pfCand_photon_hcalFraction', 'pfCand_photon_fromPV'],25)
        self.addBranches(['pfCand_electron_pt', 'pfCand_electron_eta', 'pfCand_electron_phi', 'pfCand_electron_mass', 'pfCand_electron_puppiWeight', 'pfCand_electron_hcalFraction', 'pfCand_electron_fromPV'],25)
        self.addBranches(['pfCand_muon_pt', 'pfCand_muon_eta', 'pfCand_muon_phi', 'pfCand_muon_mass', 'pfCand_muon_puppiWeight', 'pfCand_muon_hcalFraction', 'pfCand_muon_fromPV'],25)


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)

        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf]
        self.y=[alltruth]

