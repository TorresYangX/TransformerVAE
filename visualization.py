import matplotlib.pyplot as plt
import numpy as np


def visualEmdResult():
    AE =  '../small_results/AE/KDTreeAE/EMD/emd_.npy'
    VAE = '../small_results/VariationalAE/KDTreeVAE/EMD/emd_.npy'
    LCSS = '../small_results/LCSS/KDTreeLCSS/EMD/emd_.npy'
    EDR = '../small_results/EDR/KDTreeEDR/EMD/emd_.npy'
    EDwP = '../small_results/EDwP/KDTreeEDwP/EMD/emd_.npy'
    VAE_attention = '../small_results/VAE_attention/KDTreeVAE_attention/EMD/emd_.npy'
    VAE_transformer = '../small_results/VAE_transformer_new/KDTreeVAE_transformer/EMD/emd_.npy'
    VAE_nvib = '../small_results/VAE_nvib/KDTreeVAE_nvib/EMD/emd_.npy'

    AE_data = np.load(AE)
    VAE_data = np.load(VAE)
    LCSS_data = np.load(LCSS)
    EDR_data = np.load(EDR)
    EDwP_data = np.load(EDwP)
    VAE_attention_data = np.load(VAE_attention)
    VAE_transformer_data = np.load(VAE_transformer)
    VAE_nvib_data = np.load(VAE_nvib)

    # Draw a line chart

    # x axis
    x = np.arange(1, 60, 1)

    y21 = np.log(AE_data)
    y22 = np.log(VAE_data)
    # y23 = np.log(LCSS_data)
    # y24 = np.log(EDR_data)
    # y25 = np.log(EDwP_data)
    y26 = np.log(VAE_attention_data)
    y27 = np.log(VAE_transformer_data)
    y28 = np.log(VAE_nvib_data)


    # Draw a line chart
    plt.plot(x, y21, 'b-', linewidth=1.5, label='AE')
    plt.plot(x, y22, 'r-', linewidth=1.5, label='VAE')
    # plt.plot(x, y23, 'g-.', linewidth=1.5, label='LCSS')
    # plt.plot(x, y24, 'y-', linewidth=1.5, label='EDR')
    # plt.plot(x, y25, 'c-', linewidth=1.5, label='EDwP')
    plt.plot(x, y26, 'm-', linewidth=1.5, label='VAE_attention')
    plt.plot(x, y27, 'k-', linewidth=1.5, label='VAE_transformer')
    plt.plot(x, y28, 'g-', linewidth=1.5, label='VAE_nvib')
    

    # Set the x-axis label
    plt.xlabel('steps')

    # Set the y-axis label
    plt.ylabel('log(cityEMD)')

    # Set a title of the current axes.
    plt.title('log(cityEMD) of differernt models')

    # show a legend on the plot
    plt.legend()

    #save figure
    plt.savefig('../small_results/log(cityEMD).png')

    # Display a figure.
    plt.show()

if __name__ == '__main__':
    visualEmdResult()



