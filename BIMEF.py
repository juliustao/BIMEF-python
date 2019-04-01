import numpy as np
import cv2
from scipy import signal
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg

"""
img_path = '/Users/carnitas/Downloads/dark_images/0cd9b7d9-36b5-4b73-a29d-5ed319eb007d.jpg'
bgr_img = cv2.imread(img_path, 1)
b, g, r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r, g, b])     # switch it to rgb
import pdb;pdb.set_trace()
"""


def BIMEF(I, mu=0.5, k=0, a=-0.3293, b=1.1258):
    """
    :param I:   image data (of an RGB image) stored as a 3D numpy array (height x width x color)
    :param mu:  enhancement ratio
    :param k:   exposure ratio (array)
    :param a:   camera response model parameter
    :param b:   camera response model parameter
    :return:    fused: enhanced result
    """
    I = im2double(I)

    lambd = 0.5
    sigma = 5

    # t: scene illumination map
    t_b = np.amax(I, axis=2)
    t_our = cv2.resize(t_b, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)



    return #fused


def tsmooth(I, lamb=0.01, sigma=3.0, sharpness=0.001):
    I = im2double(I)
    x = I
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamb)
    return S


def computeTextureWeights(fin, sigma, sharpness):
    h1 = np.matrix(np.diff(fin, axis=1)).H
    h2 = np.matrix(fin[:, 0]).H - np.matrix(fin[:, -1]).H
    dt0_h = np.matrix(np.concatenate((h1, h2), axis=0)).H
    v1 = np.diff(fin, axis=0)
    v2 = np.expand_dims(fin[0, :] - fin[-1, :], axis=0)
    dt0_v = np.matrix(np.concatenate((v1, v2), axis=0))

    gauker_h = signal.convolve2d(dt0_h, np.rot90(np.ones((1, sigma))), mode='valid')
    gauker_v = signal.convolve2d(dt0_v, np.rot90(np.ones((sigma, 1))), mode='valid')
    W_h = np.multiply(np.absolute(gauker_h), np.absolute(dt0_h)) + sharpness
    W_h = np.divide(1, W_h)
    W_v = np.multiply(np.absolute(gauker_v), np.absolute(dt0_v)) + sharpness
    W_v = np.divide(1, W_v)

    return W_h, W_v


def solveLinearEquation(IN, wx, wy, lamb):
    r, c, ch = IN.shape
    k = r * c
    dx = -lamb * np.reshape(wx, (1, wx.size))  # row vector; in MatLab this is a column vector
    dy = -lamb * np.reshape(wy, (1, wy.size))  # row vector
    tempx = np.concatenate((wx[:, -1], wx[:, 0:-2]), axis=1)
    tempy = np.concatenate((wy[-1, :], wy[0:-2, :]), axis=0)
    dxa = -lamb * np.reshape(tempx, (1, tempx.size))  # row vector
    dya = -lamb * np.reshape(tempy, (1, tempy.size))  # row vector
    tempx = np.concatenate((wx[:, -1], np.zeros((r, c-1))), axis=1)
    tempy = np.concatenate((wy[-1, :], np.zeros((r-1, c))), axis=0)
    dxd1 = -lamb * np.reshape(tempx, (1, tempx.size))  # row vector
    dyd1 = -lamb * np.reshape(tempy, (1, tempy.size))  # row vector
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamb * np.reshape(wx, (1, wx.size))  # row vector
    dyd2 = -lamb * np.reshape(wy, (1, wy.size))  # row vector

    Ax = spdiags(np.concatenate((dxd1, dxd2), axis=0), [-k+r, -r], k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), axis=0), [-r+1, -1], k, k)
    # diagonals stored row-wise; in MatLab the diagonals are stored column-wise

    D = 1 - (dx + dy + dxa + dya)

    A = (Ax + Ay) + np.matrix(Ax + Ay).H + spdiags(D, 0, k, k)

    fast = False
    if fast:  # This method uses approximations to solve A*x=tin(:) more quickly
        L = np.linalg.cholesky(A)  # what happens if we instead use scipy's cholesky function?
        OUT = IN
        for ii in range(ch):
            tin = IN[:, :, ii]
            tout, _ = cg(A, np.reshape(tin, (tin.size, 1)), tol=0.1,
                         maxiter=50, M=np.dot(np.linalg.inv(np.matrix(L).H), np.linalg.inv(L)))
            # The conjugate gradient function in Python uses the preconditioner M, which approximates A^(-1).
            # However, MatLab uses the preconditioner M, where M approximates A. In MatLab, M = A = L * L'.
            # For Python, we calculate M = A^(-1) = L'^(-1) * L^(-1).
            OUT[:, :, ii] = np.reshape(tout, (r, c))
    else:
        OUT = IN
        for ii in range(ch):
            tin = IN[:, :, ii]
            tout = np.linalg.lstsq(A, np.reshape(tin, (tin.size, 1)))
            OUT[:, :, ii] = np.reshape(tout, (r, c))

    return OUT


def im2double(im):
    dtype = im.dtype
    if dtype is np.float64:
        return im  # do nothing if the image array has already been converted to floating points
    else:
        info = np.iinfo(dtype)  # Get the data type of the input image
        return im.astype(np.float64) / info.max  # Divide all values by the largest possible value in the datatype
