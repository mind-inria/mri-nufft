# Benchmark of MRI-NUFFT 

This are a collection of script to perform benchmarking of MRI-NUFFT operations. 

They rely on the hydra configuration package and hydra-callback for measuring statistics. (see `requirements.txt`)


To fully reproduce the  benchmarks 4 steps are necessary: 

0. Get a Cartesian Reference image file, name `cpx_cartesian.npy`
1. Generates the trajectory files  `python -m trajectory.py`
2. Run the benchmarks. Currently are available: 
 - The Performance benchmark, checking the CPU/GPU usage and memory footprint for the different backend and configuration `perf` folder.
 - The Quality benchmark that check how the pair trajectory/backend performs for the reconstruction. in `qual` folder
3. Generate some analysis figures using `perf_analysis.py`




