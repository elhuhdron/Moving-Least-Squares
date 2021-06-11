#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@update: 2020/9/25
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#from img_utils import mls_affine_deformation, mls_similarity_deformation
from img_utils import mls_rigid_deformation
from img_utils import mls_rigid_transform, create_swirl_deltas, warp_image_remap_interp
from img_utils import make_hex_xy #, _make_delta_plot

p_mr_big = np.array([
    [30, 155], [125, 155], [225, 155],
    [100, 235], [160, 235], [85, 295], [180, 293]
])
q_mr_big = np.array([
    [42, 211], [125, 155], [235, 100],
    [80, 235], [140, 235], [85, 295], [180, 295]
])

p_mona = np.array([
    [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225],
    [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
])
q_mona = np.array([
    [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225],
    [243, 225], [207, 238], [261, 237], [199, 253], [232, 281], [279, 249]
])

def show_example():
    img = plt.imread(os.path.join(sys.path[0], "double.jpg"))
    plt.imshow(img)
    plt.show()

def demo(fun, name):
    image = plt.imread(os.path.join(sys.path[0], "mr_big_ori.jpg"))
    p = p_mr_big
    q = q_mr_big

    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    plt.plot(p[:,0], p[:,1], 'rx')
    plt.title("Original Image")

    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.plot(q[:,0], q[:,1], 'rx')
    plt.title("%s Deformation \n Sampling density 1"%name)
    # transformed_image = fun(image, p, q, alpha=1, density=0.7)
    # plt.subplot(133)
    # plt.axis('off')
    # plt.imshow(transformed_image)
    # plt.title("%s Deformation \n Sampling density 0.7"%name)

    plt.tight_layout(w_pad=0.1)
    #plt.show()

def demo2(fun, name):
    image = plt.imread(os.path.join(sys.path[0], "monalisa.jpg"))
    p = p_mona
    q = q_mona

    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    plt.title("Original Image")
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.title("%s Deformation" % name)
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()

def demo3(fn, name, p=None, q=None):
    image = plt.imread(fn)

    if p is None:
        pts, deltas = create_swirl_deltas(image.shape, 20)
        p = pts; q = pts + deltas
    else:
        pts = p; deltas = p - q

    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.plot(p[:,0], p[:,1], 'rx')
    plt.title("Original Image")

    transformed_image1 = warp_image_remap_interp(image, pts, deltas)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image1, cmap='gray')
    plt.plot(q[:,0], q[:,1], 'rx')
    plt.title(name)

    plt.tight_layout(w_pad=0.1)
    #plt.show()


def demo4(fn, name, p, q):
    image = plt.imread(fn)
    img_shp = image.shape[:2]

    scl=10
    y,x = make_hex_xy(img_shp[0]//scl, img_shp[1]/scl, scl)
    v = np.concatenate((x[:,None], y[:,None]), axis=1)
    # print(v.max(0), v.min(0), v.shape)
    # _make_delta_plot(v, figno=1010)

    # compute transform in the reverse direction because
    #   the remap uses the reverse direction (dst image -> src image).
    f_v = mls_rigid_transform(v, q, p, alpha=1.0)
    pts = v; deltas = f_v - v

    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.plot(p[:,0], p[:,1], 'rx')
    plt.title("Original Image")

    transformed_image1 = warp_image_remap_interp(image, pts, deltas)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image1, cmap='gray')
    plt.plot(q[:,0], q[:,1], 'rx')
    plt.title(name)

    plt.tight_layout(w_pad=0.1)
    #plt.show()


if __name__ == "__main__":
    # demo(mls_affine_deformation, "Affine")
    # demo(mls_similarity_deformation, "Similarity")
    demo(mls_rigid_deformation, "Rigid")

    # fn = '50426488-complex-maze-puzzle-game-for-leisure-time-with-very-high-level-of-difficulty.jpeg'
    # demo3(fn, "Warp Swirl Deformation")
    fn = "mr_big_ori.jpg"
    demo3(fn, "Warp", p_mr_big, q_mr_big)

    fn = "mr_big_ori.jpg"
    demo4(fn, "Warp", p_mr_big, q_mr_big)

    plt.show()

    # demo2(mls_affine_deformation, "Affine")
    # demo2(mls_similarity_deformation, "Similarity")
    # demo2(mls_rigid_deformation, "Rigid")
