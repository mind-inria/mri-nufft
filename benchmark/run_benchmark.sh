#!/usr/bin/env bash

trajectory="trajs/trajectory_floret.bin"
datas="{n_coils:1,smaps:false} {n_coils:8,smaps:true} {n_coils:32,smaps:true}"
backend_name="finufft  gpunufft  cufinufft"
backend_upsampfac="1.25  2.0"
backend_eps="1e-3 1e-4 1e-5"

for traj in $trajectory
do
    for data in $datas
    do
        for b_name in $backend_name
        do
            for b_ups in $backend_upsampfac
            do
                for b_eps in $backend_eps
                do
                 echo python benchmark.py trajectory="$traj" data="$data" backend.name="$b_name" backend.upsampfac="$b_ups" backend.eps="$b_eps"
                done
            done
        done
    done
done
