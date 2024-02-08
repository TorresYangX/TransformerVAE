import matplotlib.pyplot as plt
import numpy as np

dataset = 'Porto'

AE =  '../results/{}/AE/KDTreeAE/EMD/emd_.npy'.format(dataset)
VAE = '../results/{}/VAE/KDTreeVAE/EMD/emd_.npy'.format(dataset)
t2vec = '../results/{}/t2vec/KDTreeT2vec/EMD/emd_.npy'.format(dataset)
NVAE = '../results/{}/NVAE/KDTreeNVAE/EMD/emd_.npy'.format(dataset)
Transformer = '../results/{}/Transformer/KDTreeTransformer/EMD/emd_.npy'.format(dataset)
LCSS = '../results/{}/LCSS/KDTreeLCSS/EMD/emd_.npy'.format(dataset)
EDR = '../results/{}/EDR/KDTreeEDR/EMD/emd_.npy'.format(dataset)
EDwP = '../results/{}/EDwP/KDTreeEDwP/EMD/emd_.npy'.format(dataset)
DTW = '../results/{}/DTW/KDTreeDTW/EMD/emd_.npy'.format(dataset)

def lineChart():

    AE_data = np.load(AE)
    VAE_data = np.load(VAE)
    t2vec_data = np.load(t2vec)
    NVAE_data = np.load(NVAE)
    Transformer_data = np.load(Transformer)
    LCSS_data = np.load(LCSS)
    EDR_data = np.load(EDR)
    EDwP_data = np.load(EDwP)
    DTW_data = np.load(DTW)

    # Draw a line chart

    # x axis
    x = np.arange(1, 60, 1)

    y21 = np.log(AE_data)
    y22 = np.log(VAE_data)
    y23 = np.log(NVAE_data)
    y24 = np.log(Transformer_data)
    y25 = np.log(LCSS_data)
    y26 = np.log(EDR_data)
    y27 = np.log(EDwP_data)
    y28 = np.log(DTW_data)
    y29 = np.log(t2vec_data)


    # Draw a line chart
    plt.plot(x, y21, 'b', linestyle='dotted', linewidth=1.5, label='AE')
    plt.plot(x, y22, 'r-', linewidth=1.5, label='VAE')
    plt.plot(x, y29, 'r.', linewidth=1, label='t2vec')
    plt.plot(x, y24, 'k-.', linewidth=1.5, label='Transformer')
    plt.plot(x, y25, 'g-.', linewidth=1.5, label='LCSS')
    plt.plot(x, y26, 'y-', linewidth=1.5, label='EDR')
    plt.plot(x, y27, 'c-.', linewidth=1.5, label='EDwP')
    plt.plot(x, y28, 'k-', linewidth=1.5, label='DTW')
    plt.plot(x, y23, 'r-', lw=2, label='NVAE')
    

    # Set the x-axis label, bold font, font size 12
    plt.xlabel('Steps', fontweight='bold', fontsize=12)

    # Set the y-axis label
    plt.ylabel('log(cityEMD)', fontweight='bold', fontsize=12)

    # Set a title of the current axes.
    plt.title('log(cityEMD) of differernt models', fontweight='bold', fontsize=12)

    # show a legend on the plot, bold font, font size 12
    plt.legend(prop={'size': 12, 'weight': 'bold'})

    #save figure
    plt.savefig('../results/{}/log(cityEMD).png'.format(dataset))

    # Display a figure.
    plt.show()
    
     

if __name__ == '__main__':
    lineChart()



