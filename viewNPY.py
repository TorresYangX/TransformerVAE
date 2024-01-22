import numpy as np

def viewNpy(targetData):
    file = np.load(targetData, allow_pickle=True)
    print(file)
    print(file.shape)
    

if __name__ == '__main__':
    targetData =  '../results/{}/KDTree{}/Evaluate_Yao/retrievedTrajectories.npy'.format('VAE', 'VAE')
    # targetData =  '../data/Experiment/groundTruth/groundTruth_8.npy'    
    viewNpy(targetData)
    
    