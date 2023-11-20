#!/usr/bin/env bash

for traj in `ls ../trajs`
do
for eps in 1e-4
do
for upsampfac in 2.0
do
for name in gpunufft
do
    python benchmark_quality.py backend.name=$name backend.eps=$eps backend.upsampfac=$upsampfac trajectory.file=../trajs/$traj
done
done
done
done
