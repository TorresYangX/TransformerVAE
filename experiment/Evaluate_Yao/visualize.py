import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AE =  '../results/AE/KDTreeAE/Evaluate_Yao/MD_NMD.csv'
VAE = '../results/VAE/KDTreeVAE/Evaluate_Yao/MD_NMD.csv'
t2vec = '../results/t2vec/KDTreeT2vec/Evaluate_Yao/MD_NMD.csv'
VAE_nvib = '../results/VAE_nvib/KDTreeVAE_nvib/Evaluate_Yao/MD_NMD.csv'
Transformer = '../results/Transformer/KDTreeTransformer/Evaluate_Yao/MD_NMD.csv'
LCSS = '../results/LCSS/KDTreeLCSS/Evaluate_Yao/MD_NMD.csv'
EDR = '../results/EDR/KDTreeEDR/Evaluate_Yao/MD_NMD.csv'
EDwP = '../results/EDwP/KDTreeEDwP/Evaluate_Yao/MD_NMD.csv'
DTW = '../results/DTW/KDTreeDTW/Evaluate_Yao/MD_NMD.csv'

AE_data = pd.read_csv(AE, header=None).to_numpy()
VAE_data = pd.read_csv(VAE, header=None).to_numpy()
t2vec_data = pd.read_csv(t2vec, header=None).to_numpy()
VAE_nvib_data = pd.read_csv(VAE_nvib, header=None).to_numpy()
Transformer_data = pd.read_csv(Transformer, header=None).to_numpy()
LCSS_data = pd.read_csv(LCSS, header=None).to_numpy()
EDR_data = pd.read_csv(EDR, header=None).to_numpy()
EDwP_data = pd.read_csv(EDwP, header=None).to_numpy()
DTW_data = pd.read_csv(DTW, header=None).to_numpy()

def matricbarChart():
    AE_NMD = float(AE_data[1][0].split(':')[1])
    VAE_NMD = float(VAE_data[1][0].split(':')[1])
    t2vec_NMD = float(t2vec_data[1][0].split(':')[1])
    VAE_nvib_NMD = float(VAE_nvib_data[1][0].split(':')[1])
    Transformer_NMD = float(Transformer_data[1][0].split(':')[1])
    LCSS_NMD = float(LCSS_data[1][0].split(':')[1])
    EDR_NMD = float(EDR_data[1][0].split(':')[1])
    EDwP_NMD = float(EDwP_data[1][0].split(':')[1])
    DTW_NMD = float(DTW_data[1][0].split(':')[1])
    
    AE_NMA = float(AE_data[2][0].split(':')[1])
    VAE_NMA = float(VAE_data[2][0].split(':')[1])
    t2vec_NMA = float(t2vec_data[2][0].split(':')[1])
    VAE_nvib_NMA = float(VAE_nvib_data[2][0].split(':')[1])
    Transformer_NMA = float(Transformer_data[2][0].split(':')[1])
    LCSS_NMA = float(LCSS_data[2][0].split(':')[1])
    EDR_NMA = float(EDR_data[2][0].split(':')[1])
    EDwP_NMA = float(EDwP_data[2][0].split(':')[1])
    DTW_NMA = float(DTW_data[2][0].split(':')[1])
    
    AE_RRNSA = float(AE_data[3][0].split(':')[1])
    VAE_RRNSA = float(VAE_data[3][0].split(':')[1])
    t2vec_RRNSA = float(t2vec_data[3][0].split(':')[1])
    VAE_nvib_RRNSA = float(VAE_nvib_data[3][0].split(':')[1])
    Transformer_RRNSA = float(Transformer_data[3][0].split(':')[1])
    LCSS_RRNSA = float(LCSS_data[3][0].split(':')[1])
    EDR_RRNSA = float(EDR_data[3][0].split(':')[1])
    EDwP_RRNSA = float(EDwP_data[3][0].split(':')[1])
    DTW_RRNSA = float(DTW_data[3][0].split(':')[1])
    
    AE_SP = float(AE_data[4][0].split(':')[1])
    VAE_SP = float(VAE_data[4][0].split(':')[1])
    t2vec_SP = float(t2vec_data[4][0].split(':')[1])
    VAE_nvib_SP = float(VAE_nvib_data[4][0].split(':')[1])
    Transformer_SP = float(Transformer_data[4][0].split(':')[1])
    LCSS_SP = float(LCSS_data[4][0].split(':')[1])
    EDR_SP = float(EDR_data[4][0].split(':')[1])
    EDwP_SP = float(EDwP_data[4][0].split(':')[1])
    DTW_SP = float(DTW_data[4][0].split(':')[1])
    
    # draw all the values in the same figure,
    x = np.arange(4)
    width = 0.1
    fig, ax = plt.subplots()
    labels = ['NMD', 'NMA', 'RRNSA', 'SP']
    ## VAE_nvib
    ax.bar(x - 2*width, [VAE_nvib_NMD, VAE_nvib_NMA, VAE_nvib_RRNSA, VAE_nvib_SP], width, label='NVAE')
    ## AE
    ax.bar(x - width, [AE_NMD, AE_NMA, AE_RRNSA, AE_SP], width, label='AE')
    ## t2vec
    ax.bar(x, [t2vec_NMD, t2vec_NMA, t2vec_RRNSA, t2vec_SP], width, label='t2vec')
    ## DTW
    ax.bar(x + width, [DTW_NMD, DTW_NMA, DTW_RRNSA, DTW_SP], width, label='DTW')
    ## EDwP
    ax.bar(x + 2*width, [EDwP_NMD, EDwP_NMA, EDwP_RRNSA, EDwP_SP], width, label='EDwP')
    ## VAE
    ax.bar(x + 3*width, [VAE_NMD, VAE_NMA, VAE_RRNSA, VAE_SP], width, label='VAE')
    ## Transformer
    ax.bar(x + 4*width, [Transformer_NMD, Transformer_NMA, Transformer_RRNSA, Transformer_SP], width, label='Transformer')
    ## EDR
    ax.bar(x + 5*width, [EDR_NMD, EDR_NMA, EDR_RRNSA, EDR_SP], width, label='EDR')
    ## LCSS
    ax.bar(x + 6*width, [LCSS_NMD, LCSS_NMA, LCSS_RRNSA, LCSS_SP], width, label='LCSS')
    
    ax.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax.set_title('Mobility Tableau', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.legend(prop={'size': 12, 'weight': 'bold'})
    plt.savefig('../results/Metric.png')
    plt.show()
    
    
def mdBarChart():
    AE_MD = float(AE_data[0][0].split(':')[1])
    VAE_MD = float(VAE_data[0][0].split(':')[1])
    VAE_nvib_MD = float(VAE_nvib_data[0][0].split(':')[1])
    Transformer_MD = float(Transformer_data[0][0].split(':')[1])
    LCSS_MD = float(LCSS_data[0][0].split(':')[1])
    EDR_MD = float(EDR_data[0][0].split(':')[1])
    EDwP_MD = float(EDwP_data[0][0].split(':')[1])
    DTW_MD = float(DTW_data[0][0].split(':')[1])
    x = np.arange(8)
    width = 0.2
    fig, ax = plt.subplots()
    ax.bar(x - width/2, [AE_MD, VAE_MD, VAE_nvib_MD, Transformer_MD, LCSS_MD, EDR_MD, EDwP_MD, DTW_MD], width, label='MD')
    ax.set_ylabel('Value')
    ax.set_title('MD')
    ax.set_xticks(x)
    ax.set_xticklabels(['AE', 'VAE', 'VAE_nvib', 'Transformer', 'LCSS', 'EDR', 'EDwP', 'DTW'])
    ax.legend()
    plt.savefig('../results/MD.png')
    plt.show()  

if __name__ == '__main__':
    matricbarChart()
    # mdBarChart()