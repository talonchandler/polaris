from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


def NCC(image, label):
    image = image / image.max()
    label = label / label.max()

    image = image.reshape(image.shape[0], image.shape[1], -1)
    label = label.reshape(label.shape[0], label.shape[1], -1)

    mean_image = np.mean(image)
    mean_label = np.mean(label)

    std_image = np.std(image)
    std_label = np.std(label)

    final_NCC = np.mean((image - mean_image) * (label - mean_label) / (std_label * std_image))

    return final_NCC


def SSIM(image, label):
    # image = (image - image.min()) / (image.max() - image.min())
    # label = (label - label.min()) / (label.max() - label.min())
    image = image / image.max()
    label = label / label.max()

    # image = (image - np.mean(image)) / np.std(image)
    # label = (label - np.mean(label)) / np.std(label)

    image = image.reshape(image.shape[0], image.shape[1], -1)
    label = label.reshape(label.shape[0], label.shape[1], -1)

    final_ssim = ssim(image, label)
    return final_ssim


def PSNR(image, label):
    image = image.reshape(image.shape[0], image.shape[1], -1)

    label = label.reshape(label.shape[0], label.shape[1], -1)

    final_psnr = psnr(image, label)
    return final_psnr


def PeakDif(spangf, labelf, BinvT, Bvertices, threshold=0.2):  # 越接近1越好
    mask = labelf[..., 0] > (threshold * np.max(labelf[..., 0]))
    spang_sh = spangf[mask]
    label_sh = labelf[mask]

    spang_odf = np.einsum('vj,pj->vp', BinvT, spang_sh)  # Radii
    spang_index = np.argmax(spang_odf, axis=0)
    spang_dirs = Bvertices[spang_index]

    label_odf = np.einsum('vj,pj->vp', BinvT, label_sh)  # Radii
    label_index = np.argmax(label_odf, axis=0)
    label_dirs = Bvertices[label_index]

    peak_cos = (spang_dirs * label_dirs).sum(axis=1)

    peak_dif = np.abs(peak_cos).mean()

    return peak_dif
