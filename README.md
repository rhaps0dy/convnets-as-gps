# Deep Convolutional Networks as shallow Gaussian Processes
Code for "Deep Convolutional Networks as shallow Gaussian Processes"
([arXiv](https://arxiv.org/abs/1808.05587)), by Adrià Garriga-Alonso, Laurence
Aitchison and Carl Edward Rasmussen. It uses
[GPflow](https://github.com/gpflow/gpflow) and
[TensorFlow](http://www.tensorflow.org/).

## Setup
This package has been tested only with python 3.5 and 3.6.

First, you need to install the package in developer mode. This will download
and install all necessary python dependencies:
```sh
cd convnets-as-gps
# optionally: pip install --user -e .
pip install -e .
```
If you have an "old" CPU, this might crash and return "Illegal Instruction". This
is because recent versions of Tensorflow come with AVX instructions enabled.
[Install tensorflow 1.5.0 to fix this](https://github.com/tensorflow/tensorflow/issues/17411#issuecomment-370393493).

## Running the experiments

**Easy way to run all the experiments**: read/run `run_all_experiments.bash`

All the experiments in the paper are run in a two-stage process:

1. Run `save_kernels.py` or `save_kernels_resnet.py`, to compute kernel matrices
  and save them to disk in a working directory. Disk space required: about 15GB
  for 1 run. Run `python3 program.py --help` for detailed information, but here
  are example invocations:
  ```sh
  python3 save_kernels.py --seed=<random seed> --n_max=200 --path=/path/to/working/directory
  python3 save_kernels_resnet.py --n_gpus=1 --n_max=200 --path=/path/to/working/directory
  ```
  In particular, the `n_max` flag determines how many training examples your GPU
  processes simultaneously. The memory requirements scale roughly proportionally
  to `n_max`^2, adjust the number for your particular hardware.
  
  You might run into "Matrix is singular" errors. In my testing, those can be
  removed by reducing `n_max`. This must be a bug of some kind in the libraries
  that I use (Tensorflow maybe?), but I have no skill or time to acquire the
  skill to troubleshoot it. Just reduce `n_max`.

2. Run `classify_gp.py` to invert the kernel matrix and calculate test results.
  This requires a lot of CPU RAM memory, at least enough to hold the matrix to
  invert with 64-bit precision. For MNIST, the main kernel matrix is ~12GB, so you
  need ~24GB of memory to maintain a decent speed. I'm sure there's a way to do
  the inverse reasonably fast and more memory-efficiently, but that would take quite a
  bit of development time.

## BibTex citation record
```bibtex
@article{garriga2018convnets,
  author  = {{Garriga-Alonso}, Adri{\`a} and Aitchison, Laurence and Rasmussen, Carl Edward},
  title   = {Deep Convolutional Networks as shallow {G}aussian Processes},
  journal = {arXiv preprint arXiv:1808.05587},
  year    = {2018},
  month   = {aug},
  url     = {https://arxiv.org/abs/1808.05587}
}
```
