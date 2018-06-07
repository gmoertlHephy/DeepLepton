'''
standardised building blocks for the models
'''

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def block_deepLeptonConvolutions(charged,neutrals,photons,electrons,muons,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Lepton convolution part. 
    '''
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)
    
    cpf=charged
    if active:
        cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf) 
        cpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)
        cpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv2')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    ppf=photons
    if active:
        ppf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv0')(ppf)
        if batchnorm:
            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm0')(ppf)
        ppf = Dropout(dropoutRate,name='ppf_dropout0')(ppf) 
        ppf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='ppf_conv1')(ppf)
        if batchnorm:
            ppf = BatchNormalization(momentum=batchmomentum,name='ppf_batchnorm1')(ppf)
        ppf = Dropout(dropoutRate,name='ppf_dropout1')(ppf)
        ppf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='ppf_conv2')(ppf)
    else:
        ppf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(ppf)

    epf=electrons
    if active:
        epf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv0')(epf)
        if batchnorm:
            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm0')(epf)
        epf = Dropout(dropoutRate,name='epf_dropout0')(epf) 
        epf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='epf_conv1')(epf)
        if batchnorm:
            epf = BatchNormalization(momentum=batchmomentum,name='epf_batchnorm1')(epf)
        epf = Dropout(dropoutRate,name='epf_dropout1')(epf)
        epf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='epf_conv2')(epf)
    else:
        epf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(epf)

    mpf=muons
    if active:
        mpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv0')(mpf)
        if batchnorm:
            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm0')(mpf)
        mpf = Dropout(dropoutRate,name='mpf_dropout0')(mpf) 
        mpf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='mpf_conv1')(mpf)
        if batchnorm:
            mpf = BatchNormalization(momentum=batchmomentum,name='mpf_batchnorm1')(mpf)
        mpf = Dropout(dropoutRate,name='mpf_dropout1')(mpf)
        mpf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='mpf_conv2')(mpf)
    else:
        mpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(mpf)

#    vtx = vertices
#    if active:
#        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
#        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
#        if batchnorm:
#            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
#        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
#        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
#    else:
#        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return npf,cpf,ppf,epf,mpf #,vtx

def block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

