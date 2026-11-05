# BCC Fe data generation and training

Run commands from `experiments/bcc_fe_3d/` with the Python environment active.
Input settings are defined in `src/settings.py`.

## System and temperatures

The EAM BCC Fe system has a 65-vacancy void. The full periodic cell, retained
region and observation core have sizes 40, 16 and 6 conventional cells per
axis, with 127,935, 8,127 and 367 reference atoms. The retained coordinate
dimension is 24,381. Small periodic MD uses the core-sized cell.

The lattice constant is 2.8665 Angstrom. The deformation is
`[[1.01, 0.05, 0], [0, 1, 0], [0, 0, 1]]`, combining tension and shear.
Training and independent validation use 50, 150, 250 and 350 K. Held-out
evaluation uses 100, 200 and 300 K interpolation and 400 K extrapolation.

## Dependencies

Install the root `requirements.txt` and `requirements-fe.txt`. The environment
must provide CUDA-enabled PyTorch, CuPy, and a LAMMPS Python interface supporting
`eam/alloy`. Test the loaders:

```bash
python -c "import torch, cupy; print(torch.cuda.is_available(), cupy.cuda.runtime.getDeviceCount())"
python -c "from lammps import lammps; s=lammps(cmdargs=['-log','none','-screen','none']); print(s.version()); s.close()"
```

MPI-enabled LAMMPS builds need their matching MPI runtime. If the loader cannot
find it, set `LD_LIBRARY_PATH` to the matching library directory. The backend
also checks the active interpreter's `lib/libmpi.so.12`, or preloads the explicit
path set by `SAM_MPI_LIBRARY`. Do not mix incompatible MPI implementations.

## 1. Prepare the external input and validate physics

```bash
python src/prepare.py
```

The script downloads the Chamati et al. Fe EAM table from the NIST URL in
`settings.py` and verifies its SHA256 before installation. The potential table
is an external runtime input and is not included in the source distribution.
For offline transfer, `python src/prepare.py --download-only` only fetches and
verifies it; alternatively provide the matching table at
`potentials/Fe_Mishin2006.eam.alloy` and run full preparation.

Full preparation checks independent EAM energy/force/stress against LAMMPS,
finite-difference force identities, the retained-region stress estimator and
buffer, then relaxes the n=6 and n=40 initial configurations. It checks small-cell
CUDA EAM parity against the independent physical evaluator. A source/protocol
identity record ties production preparation to these checks.

Preparation outputs `potentials/` and numerical records/initial configurations
in `checks/`. Production requires the preparation checks to pass.

## 2. Generate independent MD

```bash
python src/run_md.py
```

The campaign generates four training and two validation chains at each training
temperature. At all eight evaluation temperatures it generates four independent
large-reference and four small-periodic chains: 88 chains in total.

MD uses FP64 CUDA EAM and BAOAB Langevin with a 1 fs step, 0.1 ps damping,
50 ps equilibration and 200 ps initial production per chain. Coordinates,
forces and statistics are saved every 0.1 ps. Reference/small chains may extend
to 400 or 800 ps when precision or stationarity requires it. Unresolved physical
or statistical checks halt the workflow and retain the recorded configurations.
All saved MD configurations undergo the structural screening.

A subset of training temperatures may be run first:

```bash
python src/run_md.py --temperatures 50 150
# Then complete the full campaign:
python src/run_md.py
```

The pipeline acceptance record is written only after the entire campaign is
complete. All required reference data are generated locally; no precomputed
data links or acceptance records are required or distributed.

Outputs: `data/` coordinates, force labels, statistics and restarts; `checks/`
structure/convergence records; `results/` reference summaries; `logs/` and
`provenance/` local execution/source records. Rerunning unchanged jobs reuses
completed compatible outputs or resumes the GPU restart state.

## 3. Preprocess, train and select

```bash
python src/run_pipeline.py --stage train
```

The driver verifies completed dataset roles and independent seeds, runs numerical
implementation checks, and fits the shared invertible covariance preconditioner
using training frames. It then trains and selects the force model, selects the
early force-to-denoising switch, and trains/selects the switching and pure
denoising models.

Each objective has a cumulative 160,000-update budget with batch size 64,
AdamW, a cosine learning-rate schedule and EMA. The switching model inherits
raw weights, EMA, optimizer and random state without resetting the global
schedule. Checkpoint and switch selection use xx/xy stress at the four
independent validation temperatures. The selection is recomputed from the
locally generated data rather than supplied as a fitted result.

The full empirical correlated training covariance defines the Gaussian
baseline. The low-rank invertible operator is a neural preconditioner. At
400 K, endpoint means and thermal covariance/displacement scaling are used;
held-out trajectories never fit these quantities.

Outputs: `data/processed/`, `models/{objective}/`,
`results/checkpoint_selection/`, `results/switch_selection.json` and numerical
check records. All fitted parameters and checkpoints are created by this step.

## 4. Generate configurations and physical statistics

```bash
python src/run_pipeline.py --stage generate
```

The driver selects a common sampler using training temperatures and paired
comparisons against FP32/100-step Heun. It tests 25/50-step BF16/FP32 settings
and uses additional FP32 convergence checks if required. All three models and
the Gaussian baseline then generate 20,000 configurations at each evaluation
temperature from four seeds. Original EAM forces and exact bond fractions define
the physical IK stress; neural outputs do not replace the physical virial.

Outputs: `results/sampler_selection.json`, per-run samples/statistics in
`results/final/`, and `results/final_statistics.json`. Final coordinates are
retained for 200 K for the manuscript's atomic comparison. All temperatures
retain the computed statistics and metadata. The sampler validates input/source
identities and resumes at its committed boundaries without sample rejection.

`python src/run_pipeline.py --stage all` combines training and generation after
MD has completed. All child processes use the same Python interpreter.

## 5. Generate the 200 K atomic stress-contribution fields

```bash
python src/compute_mean_fields.py --workers 8
```

This optional step reads the newly generated 200 K reference, denoising and
Gaussian configurations. It assigns half of each exact configurational xx bond
contribution to each endpoint before ensemble averaging, and checks the sum
against the stored core stress. It produces per-atom means, group-based standard
errors and a saved configuration for display in `results/mean_fields_T200/`.
It performs no neural replay and requires no visualization application.

## Source groups

| Files | Responsibility |
| --- | --- |
| `settings.py`, `common.py` | Input parameters, constants, paths and metadata |
| `prepare.py`, `validate_physics.py` | Potential download, initial relaxation and physics checks |
| `physics.py`, `backend.py` | Independent EAM/IK evaluator and LAMMPS validation interface |
| `gpu_md.py`, `eam_cuda.cu` | FP64 CUDA forces and BAOAB integration |
| `md_jobs.py`, `run_md.py` | Restartable MD and independent campaign orchestration |
| `learning.py` | Preprocessing, conditional 3D CNN and training |
| `diffusion.py`, `evaluation.py` | Heun sampling, validation selection and final generation |
| `observable_parallel.py` | Ordered CPU physical-observable evaluation |
| `check_*.py`, `audit_final_structures.py` | Synthetic implementation checks and MD structure screening |
| `run_pipeline.py` | Training/generation driver |
| `compute_mean_fields.py` | Atomic xx stress-contribution data at 200 K |

The included numerical checks implement validation procedures, not saved check
results. Numerical data, model weights and runtime metadata are generated only
when these scripts run.
