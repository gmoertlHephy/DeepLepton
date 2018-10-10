#!/bin/sh

year='2016'
flavour='Muon'
short='muo'
run='_test1_'
fromrun=''
ptSelection='pt_15_to_inf'
sampleSelection='DYVsQCD'
sampleSize='mini_'
prefix=''


#Convert
#convertFromRoot.py --testdatafor /local/gmoertl/DeepLepton/DeepJet_GPU/Test/${sampleSelection}_${prefix}${flavour}${fromrun}Training/trainsamples.dc -i /local/gmoertl/DeepLepton/TrainingData/v1/${year}/${short}/${ptSelection}/${sampleSelection}/${sampleSize}test_${short}_std.txt -o /local/gmoertl/DeepLepton/DeepJet_GPU/Test/${sampleSelection}_${prefix}${flavour}${fromrun}TestData
#Predict
mini_predict.py /local/gmoertl/DeepLepton/DeepJet_GPU/Test/${sampleSelection}_${prefix}${flavour}${fromrun}Training/KERAS_model.h5 /local/gmoertl/DeepLepton/DeepJet_GPU/Test/${sampleSelection}_${prefix}${flavour}${fromrun}TestData/dataCollection.dc /local/gmoertl/DeepLepton/DeepJet_GPU/Test/${sampleSelection}_${prefix}${flavour}${run}EvaluationTestData


