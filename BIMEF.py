import numpy as np
import cv2
from scipy import signal
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg
from scipy.optimize import fminbound
from scipy.stats import entropy


def BIMEF(I, mu=0.5, k=None, a=-0.3293, b=1.1258):
    """
    :param I:   image data (of an RGB image) stored as a 3D numpy array (height x width x color)
    :param mu:  enhancement ratio
    :param k:   exposure ratio (array)
    :param a:   camera response model parameter
    :param b:   camera response model parameter
    :return:    fused: enhanced result
    """

    def maxEntropyEnhance(I, isBad=None):
        Y = rgb2gm(np.real(np.maximum(cv2.resize(I, dsize=(50, 50), interpolation=cv2.INTER_CUBIC), 0)))

        if not (isBad is None):
            isBad = cv2.resize(isBad, dsize=(50, 50), interpolation=cv2.INTER_CUBIC).T
            Y = Y[isBad]
            Y = np.reshape(Y, (Y.size, 1))

        if Y.size == 0:
            J = I
            return J

        _, opt_k, _, _ = fminbound(lambda k: -entropy(cv2.calcHist([applyK(Y, k)], [0], None, [256], [0, 1])),
                                   x1=1, x2=7)
        J = applyK(I, opt_k, a, b) - 0.01

        return J

    I = im2double(I)

    lamb = 0.5
    sigma = 5

    # t: scene illumination map
    t_b = np.amax(I, axis=2)
    t_our = cv2.resize(tsmooth(cv2.resize(t_b, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), lamb, sigma),
                       dsize=np.shape(t_b), interpolation=cv2.INTER_CUBIC)
    # We try to replicate MatLab's imresize function, which uses intercubic interpolation and anti-aliasing by default

    # k: exposure ratio
    if k is None or k.size == 0:
        isBad = t_our < 0.5  # compare t_our to 0.5 element-wise and creates a new array of truth values
        J = maxEntropyEnhance(I, isBad)
    else:
        J = applyK(I, k, a, b)
        J = np.amin(J, axis=0)

    # W: Weight Matrix
    t = np.tile(t_our, [1, 1, np.shape(I)[2]])
    W = t**mu
    I2 = I**W
    J2 = I**(1-W)
    fused = I2+J2

    return fused


def rgb2gm(I):
    if np.shape(I)[2] == 3:
        I = im2double(np.maximum(0, I))
        I = (I[:, :, 0] * I[:, :, 1] * I[:, :, 2])**(1/3)
    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = I**gamma*beta
    return J


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
