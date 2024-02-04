import matplotlib.pyplot as plt
import numpy as np
import re

def extract_float_from_string(s):
    match = re.search(r'\d+.\d+', s)
    return float(match.group()) if match else None

def visualizeSSMKNN():
    AE =  '../SSM_KNN/AE/KDTreeAE/SSM_KNN/averageRank.csv'
    VAE = '../old_model/VAE--SSM-KLweight=0/KDTreeVAE/SSM_KNN/averageRank.csv'
    NVAE = '../SSM_KNN/NVAE/KDTreeNVAE/SSM_KNN/averageRank.csv'
    NVAE_0 = '../old_model/NVAE--SSM-20epoch/KDTreeNVAE/SSM_KNN/averageRank.csv'

    #read the last line of csv file
    AE_rank = extract_float_from_string(np.loadtxt(AE, delimiter=",", skiprows=1, dtype=str)[-1])
    VAE_rank = extract_float_from_string(np.loadtxt(VAE, delimiter=",", skiprows=1, dtype=str)[-1])
    NVAE_rank = extract_float_from_string(np.loadtxt(NVAE, delimiter=",", skiprows=1, dtype=str)[-1])
    NVAE_0_rank = extract_float_from_string(np.loadtxt(NVAE_0, delimiter=",", skiprows=1, dtype=str)[-1])

    
    #draw a bar chart
    x = ['AE', 'VAE_KLweight=0', 'NVIB_KLweight=0', 'NVIB']
    y = [AE_rank, VAE_rank, NVAE_0_rank, NVAE_rank,]
    plt.bar(x, y)
    plt.xlabel('Model')
    plt.ylabel('Average Rank')
    plt.title('SSM_KNN')
    plt.savefig('../SSM_KNN/rank.png')
    plt.show()

if __name__ == '__main__':
    visualizeSSMKNN()