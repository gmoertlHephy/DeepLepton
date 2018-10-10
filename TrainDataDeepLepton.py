

from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

class TrainDataDeepLepton(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="tree"
        #self.undefTruth=['isUndefined']
        self.referenceclass='lep_isPromptId'
        self.truthclasses=['lep_isPromptId','lep_isNonPromptId','lep_isFakeId']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['lep_pt','lep_eta'])
        
        self.weightbranchX='lep_pt'
        self.weightbranchY='lep_eta'
        
        self.weight_binX = numpy.array([
                10,12.5,15,17.5,20,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves
       
    
        
from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad


class TrainData_fullTruth(TrainDataDeepLepton):
    def __init__(self):
        TrainDataDeepLepton.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['lep_isPromptId','lep_isNonPromptId','lep_isFakeId']
        if tuple_in is not None:
            prompt = tuple_in['lep_isPromptId'].view(numpy.ndarray)
            nonprompt = tuple_in['lep_isNonPromptId'].view(numpy.ndarray)
            fake = tuple_in['lep_isFakeId'].view(numpy.ndarray)
            
            return numpy.vstack((prompt,nonprompt,fake)).transpose()    
  

