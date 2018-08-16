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


class TrainData_deepLeptons_Electrons(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)

        self.addBranches([
        'lep_pt', 'lep_eta', 'lep_rho', 'lep_innerTrackChi2',
        'lep_etaSc', 'lep_sigmaIEtaIEta', 'lep_full5x5_sigmaIetaIeta', 'lep_dEtaInSeed', 'lep_dPhiScTrkIn', 'lep_dEtaScTrkIn', 'lep_eInvMinusPInv', 'lep_convVeto_float', 'lep_hadronicOverEm', 'lep_r9',
        #'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 'lep_glbTrackProbability',
        'lep_relIso03', 'lep_relIso04', 'lep_miniRelIso', 'lep_lostHits_float', 'lep_innerTrackValidHitFraction',
        'lep_jetDR', 'lep_dxy', 'lep_dz', 'lep_edxy', 'lep_edz', 'lep_ip3d', 'lep_sip3d', 'lep_EffectiveArea03', 
        'lep_jetPtRatiov1', 'lep_jetPtRelv1', 'lep_jetPtRatiov2', 'lep_jetPtRelv2', 'lep_ptErrTk', 
        'lep_npfCands_neutral_float', 'lep_npfCands_charged_float', 'lep_npfCands_photon_float', 'lep_npfCands_electron_float', 'lep_npfCands_muon_float'
        ])

        #add realtive pt, eta and phi for pf and sv!!!
        self.addBranches(['pfCand_neutral_pt', 'pfCand_neutral_puppiWeight', 'pfCand_neutral_hcalFraction', 'pfCand_neutral_fromPV', 'pfCand_neutral_dxy_pf', 'pfCand_neutral_dz_pf', 'pfCand_neutral_dzAssociatedPV'],25)
        self.addBranches(['pfCand_charged_pt', 'pfCand_charged_puppiWeight', 'pfCand_charged_hcalFraction', 'pfCand_charged_fromPV', 'pfCand_charged_dxy_pf', 'pfCand_charged_dz_pf', 'pfCand_charged_dzAssociatedPV'],50)
        self.addBranches(['pfCand_photon_pt',  'pfCand_photon_puppiWeight',  'pfCand_photon_hcalFraction',  'pfCand_photon_fromPV',  'pfCand_photon_dxy_pf',  'pfCand_photon_dz_pf',  'pfCand_photon_dzAssociatedPV' ],25)
        self.addBranches(['pfCand_electron_pt', 'pfCand_electron_dxy_pf', 'pfCand_electron_dz_pf'],25)
        self.addBranches(['pfCand_muon_pt',     'pfCand_muon_dxy_pf',     'pfCand_muon_dz_pf'    ],25)
        self.addBranches(['SV_pt', 'SV_chi2', 'SV_ndof', 'SV_dxy', 'SV_edxy', 'SV_ip3d', 'SV_eip3d', 'SV_sip3d', 'SV_cosTheta', 'SV_deltaR',
                          'SV_jetPt', 'SV_jetEta', 'SV_jetDR', 'SV_maxDxyTracks', 'SV_secDxyTracks', 'SV_maxD3dTracks', 'SV_secD3dTracks'],4)

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
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)

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
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]


class TrainData_deepLeptons_Muons(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        'lep_pt', 'lep_eta', 'lep_rho', 'lep_innerTrackChi2',
        #'lep_etaSc', 'lep_sigmaIEtaIEta', 'lep_full5x5_sigmaIetaIeta', 'lep_dEtaInSeed', 'lep_dPhiScTrkIn', 'lep_dEtaScTrkIn', 'lep_eInvMinusPInv', 'lep_convVeto_float', 'lep_hadronicOverEm', 'lep_r9',
        'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 'lep_glbTrackProbability',
        'lep_relIso03', 'lep_relIso04', 'lep_miniRelIso', 'lep_lostHits_float', 'lep_innerTrackValidHitFraction',
        'lep_jetDR', 'lep_dxy', 'lep_dz', 'lep_edxy', 'lep_edz', 'lep_ip3d', 'lep_sip3d', 'lep_EffectiveArea03', 
        'lep_jetPtRatiov1', 'lep_jetPtRelv1', 'lep_jetPtRatiov2', 'lep_jetPtRelv2', 'lep_ptErrTk', 
        'lep_npfCands_neutral_float', 'lep_npfCands_charged_float', 'lep_npfCands_photon_float', 'lep_npfCands_electron_float', 'lep_npfCands_muon_float'
        ])

        #add realtive pt, eta and phi for pf and sv!!!
        self.addBranches(['pfCand_neutral_pt', 'pfCand_neutral_puppiWeight', 'pfCand_neutral_hcalFraction', 'pfCand_neutral_fromPV', 'pfCand_neutral_dxy_pf', 'pfCand_neutral_dz_pf', 'pfCand_neutral_dzAssociatedPV'],25)
        self.addBranches(['pfCand_charged_pt', 'pfCand_charged_puppiWeight', 'pfCand_charged_hcalFraction', 'pfCand_charged_fromPV', 'pfCand_charged_dxy_pf', 'pfCand_charged_dz_pf', 'pfCand_charged_dzAssociatedPV'],50)
        self.addBranches(['pfCand_photon_pt',  'pfCand_photon_puppiWeight',  'pfCand_photon_hcalFraction',  'pfCand_photon_fromPV',  'pfCand_photon_dxy_pf',  'pfCand_photon_dz_pf',  'pfCand_photon_dzAssociatedPV' ],25)
        self.addBranches(['pfCand_electron_pt', 'pfCand_electron_dxy_pf', 'pfCand_electron_dz_pf'],25)
        self.addBranches(['pfCand_muon_pt',     'pfCand_muon_dxy_pf',     'pfCand_muon_dz_pf'    ],25)
        self.addBranches(['SV_pt', 'SV_chi2', 'SV_ndof', 'SV_dxy', 'SV_edxy', 'SV_ip3d', 'SV_eip3d', 'SV_sip3d', 'SV_cosTheta', 'SV_deltaR',
                          'SV_jetPt', 'SV_jetEta', 'SV_jetDR', 'SV_maxDxyTracks', 'SV_secDxyTracks', 'SV_maxD3dTracks', 'SV_secD3dTracks'],4)

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
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


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
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]


class TrainData_deepLeptons_Isolation(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)

        self.addBranches(['lep_pt', 'lep_eta', 'lep_rho', 'lep_npfCands_neutral_float', 'lep_npfCands_charged_float', 'lep_npfCands_photon_float', 'lep_npfCands_electron_float', 'lep_npfCands_muon_float'])
        #add realtive pt, eta and phi for pf and sv!!!
        self.addBranches(['pfCand_neutral_pt', 'pfCand_neutral_puppiWeight', 'pfCand_neutral_hcalFraction', 'pfCand_neutral_fromPV', 'pfCand_neutral_dxy_pf', 'pfCand_neutral_dz_pf', 'pfCand_neutral_dzAssociatedPV'],25)
        self.addBranches(['pfCand_charged_pt', 'pfCand_charged_puppiWeight', 'pfCand_charged_hcalFraction', 'pfCand_charged_fromPV', 'pfCand_charged_dxy_pf', 'pfCand_charged_dz_pf', 'pfCand_charged_dzAssociatedPV'],50)
        self.addBranches(['pfCand_photon_pt',  'pfCand_photon_puppiWeight',  'pfCand_photon_hcalFraction',  'pfCand_photon_fromPV',  'pfCand_photon_dxy_pf',  'pfCand_photon_dz_pf',  'pfCand_photon_dzAssociatedPV' ],25)
        self.addBranches(['pfCand_electron_pt', 'pfCand_electron_dxy_pf', 'pfCand_electron_dz_pf'],25)
        self.addBranches(['pfCand_muon_pt',     'pfCand_muon_dxy_pf',     'pfCand_muon_dz_pf'    ],25)
        self.addBranches(['SV_pt', 'SV_chi2', 'SV_ndof', 'SV_dxy', 'SV_edxy', 'SV_ip3d', 'SV_eip3d', 'SV_sip3d', 'SV_cosTheta', 'SV_deltaR',
                          'SV_jetPt', 'SV_jetEta', 'SV_jetDR', 'SV_maxDxyTracks', 'SV_secDxyTracks', 'SV_maxD3dTracks', 'SV_secD3dTracks'],4)


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
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


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
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]
