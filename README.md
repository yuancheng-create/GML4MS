# GML4MS

Companion code for the paper:

**[Generative marginal learning for finite-size molecular simulation with application to stress calculation](https://arxiv.org/abs/2610.25105)**

Cheng Yuan (2026), arXiv:2610.25105. [Read the PDF](https://arxiv.org/pdf/2610.25105).

This repository provides training and data-generation workflows for generative
marginal learning (GML), including a two-dimensional Lennard-Jones (LJ) crystal
with seven vacancies and a three-dimensional BCC iron crystal with a 65-vacancy
void. The guides below describe how to generate the required data, train the
models and sample configurations locally.

This distribution contains English documentation, Python/CUDA source, dependency
lists and Git text settings. All input parameters are defined in Python source.
MD trajectories, processed arrays, fitted parameters, trained weights, generated
samples, result tables, figures and logs are produced locally and are excluded
from the distribution. The external EAM potential is downloaded and verified at
runtime. [SOURCE_FILES.txt](SOURCE_FILES.txt) lists the complete public file set.

## Repository structure

```text
.
├── README.md
├── SOURCE_FILES.txt
├── requirements.txt            # Common dependencies
├── requirements-fe.txt         # Additional dependencies for BCC Fe
├── docs/
│   ├── lj.md                   # LJ generation, preprocessing, training, sampling
│   └── fe3d.md                 # BCC Fe preparation, MD, training, sampling
└── experiments/
    ├── lj/
    │   ├── common.py           # LJ input settings and local paths
    │   ├── prepare.py          # Initial lattice and relaxation
    │   ├── md_core.py          # Force-shifted LJ and neighbor kernel
    │   ├── md_data.py, md.py    # MD integration and independent data-generation jobs
    │   ├── preprocessing.py    # Training-only whitening and force-score labels
    │   ├── network.py          # Conditional physical CNN
    │   ├── training.py         # Force, denoising and joint objectives
    │   ├── diffusion.py        # Probability-flow Heun integrator
    │   └── generation.py       # Gaussian and learned configuration generation
    └── bcc_fe_3d/
        ├── run_python.sh       # Optional active-interpreter wrapper
        └── src/
            ├── settings.py    # BCC Fe input settings
            ├── prepare.py     # EAM download and numerical preparation
            ├── physics.py, backend.py
            ├── gpu_md.py, eam_cuda.cu
            ├── md_jobs.py, run_md.py
            ├── learning.py, diffusion.py, evaluation.py
            ├── run_pipeline.py
            └── compute_mean_fields.py
```

The experiment directories also contain local README files, the physical
observable implementations, and numerical checks used by the workflows.

| System | Training and validation temperatures | Held-out temperatures | Objectives |
| --- | --- | --- | --- |
| LJ | 0.02, 0.04, 0.06, 0.08, 0.10 (reduced units) | 0.03, 0.05, 0.07, 0.09 | Force, denoising, joint |
| BCC Fe | 50, 150, 250, 350 K | 100, 200, 300, 400 K | Force, denoising, force-to-denoising |

The 400 K BCC Fe case is extrapolation. Training, validation and reference MD
use independent trajectories. Held-out data do not enter preprocessing or model
and sampler selection.

## Environment

Use Linux or WSL2 and Python 3.12. Neural training and sampling require an NVIDIA
GPU and CUDA-enabled PyTorch. LJ MD uses CPU/Numba. BCC Fe production MD uses
CuPy and the supplied FP64 CUDA EAM kernel; LAMMPS is used for reference checks
and initial relaxation.

Create an environment, for example with uv:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

An existing compatible research environment can be activated instead. The CUDA
check must print `True` before neural training or sampling. For BCC Fe, additionally
install:

```bash
python -m pip install -r requirements-fe.txt
```

LAMMPS must provide the Python module, shared library and `eam/alloy` pair style.
MPI-enabled builds require their matching MPI runtime. See the [3D setup](docs/fe3d.md)
for loader checks. TeX, notebook servers and visualization applications are not
required by these workflows.

The lightweight validation environment used Python 3.12.15, NumPy 2.5.3,
SciPy 1.18.1, Numba 0.68.0, PyTorch 2.11.0+cu128 and CuPy 14.2.0. The dependency
ranges are not an exhaustive compatibility matrix. Full production MD and full
training have not been rerun as part of preparing this distribution; the
LAMMPS-dependent checks must pass during initial 3D preparation.

## Run the experiments

Follow the complete guides in order:

1. [LJ workflow](docs/lj.md): relax the two cells, generate independent MD,
   preprocess, train the three models, and generate configurations.
2. [BCC Fe workflow](docs/fe3d.md): download/verify EAM, validate physics, generate
   all required MD, train/select the three models, then generate configurations
   and optionally the 200 K atomic stress-contribution fields.

These are full research runs. Allow substantial GPU memory, CPU time and disk
space. Scripts resolve files relative to their source directory. Run the two
experiments in separate Python processes because they use local modules with
common names. Shell wrappers use the active `python`, or `SAM_PYTHON` when set;
3D subprocesses use the launching interpreter.

## Outputs and resumption

All runtime arrays, weights and metadata are written under the corresponding
experiment directory and ignored by Git. The guides identify the output paths.
Rerun the same command with unchanged source and inputs to resume at its
supported boundary. LJ MD restarts unfinished jobs; neural training, neural
sampling and 3D GPU MD use their recorded checkpoint boundaries.

Do not run simultaneous writers to the same output paths. For a changed
protocol, use another checkout rather than mixing outputs. Failed configurations
are retained where supported; generated configurations are not filtered to
improve agreement.

## Get the code

Clone the repository, then follow the environment setup and experiment guides:

```bash
git clone https://github.com/yuancheng-create/GML4MS.git
cd GML4MS
```

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@misc{yuan2026generativemarginallearning,
  title         = {Generative marginal learning for finite-size molecular simulation with application to stress calculation},
  author        = {Cheng Yuan},
  year          = {2026},
  eprint        = {2610.25105},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  doi           = {10.48550/arXiv.2610.25105},
  url           = {https://arxiv.org/abs/2610.25105}
}
```
