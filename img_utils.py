#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares.

    * Affine deformation
    * Similarity deformation
    * Rigid deformation

For more details please reference the documentation:

    Moving-Least-Squares/doc/Image Deformation.pdf

or the original paper:

    Image deformation using moving least squares
    Schaefer, Mcphail, Warren.

Note:
    In the original paper, the author missed the weight w_j in formular (5).
    In addition, all the formulars in section 2.1 miss the w_j.
    And I have corrected this point in my documentation.

@author: Jarvis ZHANG
@date: 2017/8/8
@update: 2020/9/25
"""

import numpy as np
#from skimage.transform import rescale

import scipy.ndimage as nd
import scipy.interpolate as interp

np.seterr(divide='ignore', invalid='ignore')

def mls_affine_deformation_1pt(p, q, v, alpha=1):
    ''' Calculate the affine deformation of one point.
    This function is used to test the algorithm.
    '''
    ctrls = p.shape[0]
    np.seterr(divide='ignore')
    w = 1.0 / np.sum((p - v) ** 2, axis=1) ** alpha
    w[w == np.inf] = 2**31-1
    pstar = np.sum(p.T * w, axis=1) / np.sum(w)
    qstar = np.sum(q.T * w, axis=1) / np.sum(w)
    phat = p - pstar
    qhat = q - qstar
    reshaped_phat1 = phat.reshape(ctrls, 2, 1)
    reshaped_phat2 = phat.reshape(ctrls, 1, 2)
    reshaped_w = w.reshape(ctrls, 1, 1)
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)
    try:
        inv_pTwp = np.linalg.inv(pTwp)
    except np.linalg.linalg.LinAlgError:
        if np.linalg.det(pTwp) < 1e-8:
            new_v = v + qstar - pstar
            return new_v
        else:
            raise
    mul_left = v - pstar
    mul_right = np.sum(reshaped_phat1 * reshaped_w * qhat[:, np.newaxis, :], axis=0)
    new_v = np.dot(np.dot(mul_left, inv_pTwp), mul_right) + qstar
    return new_v

def mls_affine_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Affine deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)                 # [2, 2, grow, gcol]
    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = reshaped_w * reshaped_phat1                                             # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right =mul_right.transpose(0, 3, 4, 1, 2)                              # [ctrls, grow, gcol, 2, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)           # [ctrls, grow, gcol, 1, 1]
    reshaped_A = A.reshape(ctrls, 1, grow, gcol)                                        # [ctrls, 1, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = np.sum(reshaped_A * qhat, axis=0) + qstar                            # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                        (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image


def mls_similarity_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2),
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)        # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

     # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    transformers = reshaped_temp / reshaped_mu  + qstar                                 # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                        (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image


def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)         # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)           # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar  + qstar            # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                        (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image

def mls_rigid_transform(v, p, q, alpha=1.0):
    ''' Rigid transform
    ### Params:
        * v - ndarray: an array with size [m, 2], points to transform
        * p - ndarray: an array with size [n, 2], control points
        * q - ndarray: an array with size [n, 2], deformed control points
        * alpha - float: parameter used by weights
    ### Return:
        f_r - ndarray: an array with size [m, 2], deformed points from v
    '''

    m, n = v.shape[0], p.shape[0]
    assert( q.shape[0] == n )

    # shape (n,m,2)
    p_ij = p[:,None,:]
    q_ij = q[:,None,:]
    v_ij = v[None,:,:]

    # weights, shape (n,m)
    w = 1.0 / np.sum((p_ij - v_ij)**2, axis=2)**alpha
    w_ij = w[:,:,None]
    sum_w = np.sum(w, axis=0)
    #sum_w_ij = sum_w[None,:,None]

    # centroids
    p_star = (w_ij * p_ij).sum(0) / sum_w[:,None] # shape (m,2)
    q_star = (w_ij * q_ij).sum(0) / sum_w[:,None] # shape (m,2)
    p_star_ij = p_star[None,:,:]
    q_star_ij = q_star[None,:,:]
    p_hat_ij = p_ij - p_star_ij
    q_hat_ij = q_ij - q_star_ij

    # similarity transform solved matrix
    p_hat_ij_x = np.zeros_like(p_hat_ij)
    # from paper: (_x) is an operator on 2D vectors such that (x, y) = (−y, x).
    p_hat_ij_x[:,:,0] = -p_hat_ij[:,:,1]
    p_hat_ij_x[:,:,1] = p_hat_ij[:,:,0]
    v_minus_p_star = v_ij - p_star_ij
    v_minus_p_star_x = np.zeros_like(v_minus_p_star)
    v_minus_p_star_x[:,:,0] = -v_minus_p_star[:,:,1]
    v_minus_p_star_x[:,:,1] = v_minus_p_star[:,:,0]
    A_ij = w_ij[:,:,:,None] * np.matmul(\
            np.concatenate((p_hat_ij[:,:,:,None], -p_hat_ij_x[:,:,:,None]), axis=3),
            np.concatenate((v_minus_p_star[:,:,:,None], -v_minus_p_star_x[:,:,:,None]), axis=3).transpose(0,1,3,2),
            )

    # f_r, shape (m,2)
    f_bar_r = np.matmul(q_hat_ij[:,:,None,:], A_ij).sum(0).reshape(m,2)
    v_minus_p_star = v_minus_p_star.reshape(m,2)
    f_r = np.sqrt((v_minus_p_star**2).sum(1))[:,None] * f_bar_r / np.sqrt((f_bar_r**2).sum(0))[:,None] + q_star

    return f_r

def create_swirl_deltas(img_shape, warp_rs):
    spacing = np.pi/(min(img_shape)/2.5)
    nimages = len(warp_rs)

    #x,y = np.mgrid[:img_shape[0]:50,:img_shape[1]:50]
    x,y = np.meshgrid(np.linspace(0,img_shape[0]-1,img_shape[0]//20),
                      np.linspace(0,img_shape[1]-1,img_shape[1]//20))
    x = x.flat[:]; y = y.flat[:]
    xc = x - x.mean(0); yc = y - y.mean(0)
    pts = np.concatenate((x[:,None], y[:,None]), axis=1)

    deltas = [None]*nimages
    for i in range(nimages):
        deltas[i] = warp_rs[i]*np.concatenate((np.sin(yc*spacing)[:,None], -np.sin(xc*spacing)[:,None]), axis=1)

    return pts, deltas

def warp_image_remap_interp(img, pts, deltas):
    img_shape = img.shape
    # img_dtype = img.dtype

    # create a series of warped images
    grid_x, grid_y = np.indices((img_shape[0], img_shape[1]), dtype=np.double)

    vx = interp.griddata(pts, deltas[:,0], (grid_x, grid_y), fill_value=0., method='cubic')
    vy = interp.griddata(pts, deltas[:,1], (grid_x, grid_y), fill_value=0., method='cubic')
    coords = [vx+grid_x, vy+grid_y]
    image = nd.map_coordinates(img, coords, order=1, mode='constant', cval=0.0, prefilter=False)

    rdeltas = -deltas
    vx = interp.griddata(pts, rdeltas[:,0], (grid_x, grid_y), fill_value=0., method='cubic')
    vy = interp.griddata(pts, rdeltas[:,1], (grid_x, grid_y), fill_value=0., method='cubic')
    coords = [vx+grid_x, vy+grid_y]
    rimage = nd.map_coordinates(image, coords, order=1, mode='constant', cval=0.0, prefilter=False)

    return image, rimage
