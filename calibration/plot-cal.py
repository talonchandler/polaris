import matplotlib.pyplot as plt
import tifffile
import numpy as np
from polaris.micro import multi
from polaris import spang

def abc2theta(abc, theta):
    return abc[0] + abc[1]*np.cos(2*theta) + abc[2]*np.sin(2*theta)

def get_cal_data(filename, spatial_buffer=225):
    print('Reading ' + filename)
    with tifffile.TiffFile(filename) as tf:
        data_temp = tf.asarray() # ZYX order
        print("Size = ", data_temp.shape)
    print("Spatial buffer = ", spatial_buffer)        
    data = np.zeros((210,) + data_temp.shape[1:3])
    data[0:data_temp.shape[0], 0:data_temp.shape[1], 0:data_temp.shape[2]] = data_temp
    data = np.swapaxes(data, 0, 2)
    data = data.reshape(data.shape[0:2]+ (7, 30))
    sb = spatial_buffer
    pb = 3 # polarization buffer
    data_sub = data[sb:-sb, sb:-sb, :, pb:-pb]
    mean = np.mean(data_sub, axis=(0,1,3))
    std = np.std(data_sub, axis=(0,1,3))
    return np.stack((mean, std)).T

f, axs = plt.subplots(1, 2, figsize=(10, 4))

# # Read in data and save means
# # (uncomment this block to read from ./LS_488/ or elsewhere)
# row_filenames = ['./LS_488/']
# col_filenames = ['SPIMA/SPIMA', 'SPIMB/SPIMB']
# line_filenames = ['_Tilt_0_0.tif', '_Tilt_1_0.tif', '_Tilt_-1_0.tif']
# sbs = [225, 125]
# data = np.zeros((len(row_filenames), len(col_filenames), len(line_filenames), 7, 2))
# for i in range(len(row_filenames)):
#     for j, ax in enumerate(axs):
#         for k in range(3):
#             filename = row_filenames[i]+col_filenames[j]+line_filenames[k]
#             data[i,j,k,:,:] = get_cal_data(filename, spatial_buffer=sbs[j])
# np.save('mean-data.npy', data)

# Load data
data = np.load('mean-data.npy')

colors = ['g', 'b', 'r']
labels = ['Tilt 0', 'Tilt 1', 'Tilt -1']

i = 0
for j in range(2):
    for k in range(3):
        ax = axs[j]

        x0 = np.linspace(0,180,1000)
        x0t = np.array(np.deg2rad(x0))
        x1 = [0,45,60,90,120,135,180]
        x1t = np.array(np.deg2rad(x1))

        y = data[0,j,k,:,0]

        # Least squares fit
        A = np.zeros((len(x1t), 3))
        A[:,0] = 1
        A[:,1] = np.cos(2*x1t)
        A[:,2] = np.sin(2*x1t)
        abc = np.linalg.lstsq(A, y)[0]
        y_lst = np.array([abc2theta(abc, np.deg2rad(xx)) for xx in x0])

        # ax.plot(x1, y, 'o'+colors[k], label=labels[k]) # Plot dots
        ax.errorbar(x1, y, yerr=data[i,j,k,:,1], fmt='o'+colors[k], label=labels[k]) # Plot dots        
        ax.plot(x0, y_lst, '-'+colors[k])

        print('Pol offset: ' + str(np.round(x0[np.argmax(y_lst)], 4)))

    # SPIMA
    if j == 0: # SPIMA
        mod_depth = 0.179
    else:
        mod_depth = 0.260

    mean = np.mean(data[i,j,:,:,0])
    ax.plot(x0, mod_depth*mean*np.cos(2*np.pi*(x0-11)/180) + mean, ':k', label='Model prediction')

    # Labels
    ax.set_xlabel('Excitation polarization $\\hat{\\mathbf{p}}$ (degrees)')
    ax.set_ylabel('Lake response $H_0^0(0, \\hat{\\mathbf{p}})$')
    ax.set_xlim([-10,190])
    ax.xaxis.set_ticks([0, 45, 90, 135, 180])
    ax.legend(frameon=False)

axs[0].annotate('SPIMA', xy=(0,0), xytext=(0.5, 1.1), xycoords='axes fraction', textcoords='axes fraction', ha='center', va='center')
axs[1].annotate('SPIMB', xy=(0,0), xytext=(0.5, 1.1), xycoords='axes fraction', textcoords='axes fraction', ha='center', va='center')


f.savefig('calibration-488.pdf', bbox_inches='tight')
