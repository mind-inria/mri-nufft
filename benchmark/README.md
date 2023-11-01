# Benchmark of MRI-NUFFT 

This are a collection of script to perform benchmarking of MRI-NUFFT operations. 

They rely on the hydra configuration package and hydra-callback for measuring statistics. 

Configuration of the benchmark are available in the conf directory. 

To fully reproduce this benchmark 4 steps are necessary: 

1. Generates the trajectory files  `python -m 01_ generate_trajectories.py`
2. Run the benchmark scheduler with `python -m 02_benchmark.py --multirun +trajectory=traj/*.bin +backend=glob(*)`
3. Gather and preprocess the benchmark files `python -m 03_preprocessing.py`
4. Plot the results `python -m 04_plot_results.py`



