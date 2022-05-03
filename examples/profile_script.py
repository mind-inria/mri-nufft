#!/usr/bin/env python
"""Basic Profiling script, to be used with NVidia Profiling tools."""
import argparse

import numpy as np
import cupy as cp

from mriCufinufft import MRICufiNUFFT


def rand11(size):
    """Return a uniformly random array in [-1,1]."""
    return np.random.uniform(low=-1.0, high=1, size=size)


parser = argparse.ArgumentParser()
parser.add_argument("shape", nargs="+", type=int)
parser.add_argument("action", type=str, choices=("op", "adj", "both"))
parser.add_argument("device", type=str, choices=("H", "D"), default="H")
parser.add_argument("--coils", type=int, default=1)
parser.add_argument("--smaps", action="store_true", default=False)
parser.add_argument("--cached", action="store_true", default=False)
parser.add_argument("--plans", type=str, dest='plans')
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--ratio", type=float, default=1.0,
                    help="sampling ratio #Upts/#NUpts")
parser.add_argument("--nodensity", dest="density",
                    action="store_false", default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.shape)
    n_s = int(np.prod(args.shape) * args.ratio)
    shape = args.shape
    n_c = args.coils
    if args.smaps:
        smaps = 1j * np.random.randn(n_c, *shape)
        smaps += np.random.randn(n_c, *shape)
        smaps = smaps / np.linalg.norm(smaps, axis=0)
        smaps = smaps.astype(np.complex64)
    else:
        smaps = None
    samples = rand11((n_s, len(shape))).astype(np.float32) * np.pi
    obj = MRICufiNUFFT(samples,
                       shape,
                       n_coils=n_c,
                       smaps=smaps,
                       smaps_cached=args.cached,
                       plan_setup=args.plans,
                       density=args.density,
                       eps=args.eps)

    if args.action in ["op", "both"]:
        n_coils_img = n_c
        if args.smaps:
            n_coils_img = 1
        image_data = np.random.randn(n_coils_img, *shape) + \
            1j * np.random.randn(n_coils_img, *shape)
        image_data = np.squeeze(image_data)
        image_data = image_data.astype(np.complex64)
        image_data = np.ascontiguousarray(image_data)
        if args.device == "D":
            image_data = cp.asarray(image_data)
            image_data = cp.ascontiguousarray(image_data)
        obj.op(image_data)

    if args.action in ["adj", "both"]:
        kspace_data = np.squeeze(
            np.random.randn(n_c, n_s) + 1j * np.random.randn(n_c, n_s))
        kspace_data = kspace_data.astype(np.complex64)
        kspace_data = np.ascontiguousarray(kspace_data)
        if args.device == "D":
            kspace_data = cp.asarray(kspace_data)

        obj.adj_op(kspace_data)
    print(type(obj.raw_op.opts[1]))
    print("repr: ", repr(obj.raw_op.opts[1]))
    print("str: ", str(obj.raw_op.opts[1]))
