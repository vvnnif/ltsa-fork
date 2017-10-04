""" test ltsa package imports """

import ltsa
import h5py
import numpy as np

if __name__ == "__main__":

    fTr = h5py.File('baseline/Train_N400.mat')
    Output = np.reshape(fTr['output'].value.T, [400, 26 * 26 * 26])
    OutputR, dictOut = ltsa.utils.preprocessing.pre(Output)

    ManifoldModel = ltsa.LocalTangentSpaceAlignment(OutputR, 25, 5)
    ManifoldModel.solve()

    Ypred = ManifoldModel.pre_image(ManifoldModel.t)

    print np.mean((OutputR-Ypred)**2)
