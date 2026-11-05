# BCC Fe under tension and shear

Training/validation: 50, 150, 250, 350 K. Held-out evaluation: 100, 200, 300, 400 K.
Input settings are defined in `src/settings.py`.

After installing the root 3D dependencies, run from this directory:

```bash
python src/prepare.py
python src/run_md.py
python src/run_pipeline.py --stage train
python src/run_pipeline.py --stage generate
# Optional manuscript atomic stress-contribution comparison:
python src/compute_mean_fields.py --workers 8
```

See [the full BCC Fe workflow](../../docs/fe3d.md) for MPI/LAMMPS setup,
preconditions, outputs and resumption.
