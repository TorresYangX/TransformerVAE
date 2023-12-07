import matplotlib.pyplot as plt
import numpy as np


def visualEmdResult():
    AE =  '../results/AE/KDTreeAE/EMD/emd_.npy'
    VAE = '../results/VAE/KDTreeVAE/EMD/emd_.npy'
    VAE_nvib = '../results/VAE_nvib/KDTreeVAE_nvib/EMD/emd_.npy'
    Transformer = '../results/Transformer/KDTreeTransformer/EMD/emd_.npy'
    LCSS = '../results/LCSS/KDTreeLCSS/EMD/emd_.npy'
    EDR = '../results/EDR/KDTreeEDR/EMD/emd_.npy'
    EDwP = '../results/EDwP/KDTreeEDwP/EMD/emd_.npy'
    DTW = '../results/DTW/KDTreeDTW/EMD/emd_.npy'

    AE_data = np.load(AE)
    VAE_data = np.load(VAE)
    VAE_nvib_data = np.load(VAE_nvib)
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
    y23 = np.log(VAE_nvib_data)
    y24 = np.log(Transformer_data)
    y25 = np.log(LCSS_data)
    y26 = np.log(EDR_data)
    y27 = np.log(EDwP_data)
    y28 = np.log(DTW_data)


    # Draw a line chart
    plt.plot(x, y21, 'b-', linewidth=1.5, label='AE')
    plt.plot(x, y22, 'r-', linewidth=1.5, label='VAE')
    plt.plot(x, y23, 'm-', linewidth=1.5, label='VAE_nvib')
    plt.plot(x, y24, 'k-', linewidth=1.5, label='Transformer')
    plt.plot(x, y25, 'g-.', linewidth=1.5, label='LCSS')
    plt.plot(x, y26, 'y-', linewidth=1.5, label='EDR')
    # plt.plot(x, y27, 'c-', linewidth=1.5, label='EDwP')
    # plt.plot(x, y28, 'k-', linewidth=1.5, label='DTW')
    

    # Set the x-axis label
    plt.xlabel('steps')

    # Set the y-axis label
    plt.ylabel('log(cityEMD)')

    # Set a title of the current axes.
    plt.title('log(cityEMD) of differernt models')

    # show a legend on the plot
    plt.legend()

    #save figure
    plt.savefig('../results/log(cityEMD).png')

    # Display a figure.
    plt.show()

if __name__ == '__main__':
    visualEmdResult()



