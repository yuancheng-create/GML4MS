# LJ data generation and training

Run the following commands from `experiments/lj/` with the Python environment
active. The system and temperatures are fixed to the manuscript's seven-vacancy
LJ example. Numerical settings are defined in `common.py`.

## System

The large periodic cell has 48 x 48 triangular-lattice sites, with the central
atom and six nearest neighbors removed: 2,297 atoms. The retained region has
317 atoms and 634 coordinates. The observation region and small periodic cell
have size n=8; the small cell contains 57 atoms. The force-shifted LJ cutoff is
2.5, the lattice spacing is 1.1161533029896453, and reduced units use
m = epsilon = sigma_LJ = k_B = 1.

Training/validation temperatures are 0.02, 0.04, 0.06, 0.08 and 0.10. Evaluation
uses all nine temperatures from 0.02 through 0.10 in increments of 0.01.

## 1. Prepare and generate MD

```bash
python prepare.py --vacancies 7
python md.py --pilot --workers 4
python md.py --production --workers 4
python md.py --extend --workers 4
```

Preparation creates and relaxes n=8 and n=48 lattices. The pilot checks the
labelled solid basin at the temperature endpoints; production requires its
stability gate to pass.

MD uses BAOAB Langevin, dt=0.002, gamma=1, 100,000 equilibration steps and
1,000,000 production steps per segment, saving every 100 steps. At each training
temperature there are six training, two validation and four test trajectories.
Held-out temperatures have four independent test trajectories. Each evaluation
temperature also has four small periodic trajectories.

`--extend` continues the six training chains from their saved coordinates with
fresh velocities and equilibration. Together the segments provide 120,000
training frames per temperature; validation provides 20,000 frames and each MD
comparison provides 40,000 reference frames per temperature.

Outputs are in `cases/vacancy_7/initial/`, `pilot/`, `data/` and their associated
geometry, stability and metadata files. An unfinished LJ MD job reruns that job;
it does not have a per-step restart. Completed jobs are reused only with matching
parameters.

## 2. Preprocess

```bash
python preprocessing.py --vacancies 7
```

Run this after both production and extension. Preprocessing estimates means and
regularized full covariance matrices from training data only, whitens coordinates
and constructs aligned full-system force-score labels. At held-out temperatures,
means and covariance divided by temperature are interpolated between training
temperatures. Validation data do not fit the transformation.

Output: `cases/vacancy_7/processed/`, including training/validation coordinate,
label and temperature arrays, Gaussian parameters and diagnostics. There are
600,000 training and 100,000 validation frames in total. Array construction
requires sufficient RAM and disk space.

## 3. Train the three objectives

```bash
python training.py --vacancies 7 --objective force --seed 0
python training.py --vacancies 7 --objective noise --seed 0
python training.py --vacancies 7 --objective joint --seed 0
```

`noise` denotes denoising. Joint training uses 0.5 times the force loss plus
4 times the denoising loss. The physical CNN, initialization and training draws
are shared across objectives. Each model runs 100,000 updates with batch size
256, AdamW and EMA.

Output: `cases/vacancy_7/models/{objective}_seed0/`. `last.pt` contains resumable
optimizer/RNG state; `final.pt` contains the final state; `step_006000.pt` supports
the manuscript's separate 6,000-update comparison. These are generated model
files and are not included in this distribution.

## 4. Generate configurations

```bash
python generation.py --vacancies 7 --method gaussian --seed 0
python generation.py --vacancies 7 --method force --seed 0
python generation.py --vacancies 7 --method noise --seed 0
python generation.py --vacancies 7 --method joint --seed 0
```

Each command generates 50,000 configurations at each evaluation temperature.
Neural generation uses the final EMA, 25 Heun steps, BF16 network evaluations,
FP32 state and FP64 restored coordinates/observables. A paired implementation
check records agreement of graph/plain FP32 sampling and the difference from a
400-step FP32 reference. Gaussian generation uses the fitted correlated
covariance; this implementation uses CUDA for its random draws too.

Output: `cases/vacancy_7/generated/{method}_seed0_T*/`, including coordinates,
physical stress/RDF statistics, progress and metadata. Sampling resumes at
committed chunks with unchanged parameters and checkpoint identity. All generated
configurations enter the statistics without rejection or relaxation.

For the 6,000-update comparison, use the separately saved checkpoint:

```bash
python generation.py --vacancies 7 --method force --seed 0 --checkpoint-step 6000
python generation.py --vacancies 7 --method noise --seed 0 --checkpoint-step 6000
python generation.py --vacancies 7 --method joint --seed 0 --checkpoint-step 6000
```

These outputs go to `cases/vacancy_7/generated_step6000/`. The Gaussian and MD
baselines are shared with the final-checkpoint comparison.

## Source files

| Files | Purpose |
| --- | --- |
| `common.py` | Input settings, paths and atomic metadata writes |
| `prepare.py`, `md_core.py` | Seven-vacancy geometry, relaxation, LJ forces and neighbors |
| `md_data.py`, `md.py` | BAOAB integration and independent MD jobs |
| `preprocessing.py` | Training-only means, covariance, whitening and force labels |
| `network.py`, `training.py` | Conditional physical CNN and training objectives |
| `diffusion.py`, `generation.py` | Heun integration and configuration generation |
| `observables.py` | Spatial IK stress and pair-structure statistics |

All arrays, checkpoint files, metadata and diagnostics are generated by the
commands above. The code does not read precomputed manuscript results.
