# MLflow-tracked PCR-GLOBWB pipeline

A thin orchestration layer (ported from [plumber](https://github.com/aurelio-raffa/plumber)) that runs a
whole PCR-GLOBWB experiment as **one MLflow-tracked pipeline**, submittable as a single LSF job:

1. **setup** — create the output / working directories
2. **compute_basins** *(optional)* — compute LDD basins/tiles and the per-tile clone & landmask maps
3. **create_ini** — instantiate a concrete `.ini` from a template
4. **run_model** — launch PCR-GLOBWB (parallel/tiled or serial)

Each step is a normal Python script (a [Fire](https://github.com/google/python-fire) CLI) under `src/stages/`
that **reuses the existing root tools** (`compute_ldd_basins.py`, `create_tile_clone_maps.py`,
`create_ini_config.py`, `model/*_runner*.py`) — the harness only sequences and tracks them.

## How it runs

```
run_project.py  ──mlflow.projects.run──▶  src/stages/run.py  (orchestrator: 1 parent MLflow run)
                                              │   for each stage:  mlflow.run ─▶ python src/stages/<stage>.py --k v
                                              ├─ setup.py          → os.makedirs(...)
                                              ├─ compute_basins.py → compute_ldd_basins.py → create_tile_clone_maps.py
                                              ├─ create_ini.py     → create_ini_config.py  → moves INI to output-path
                                              └─ run_model.py      → model/parallel_pcrglobwb_runner*.py | deterministic_runner.py
```

The orchestrator opens **one parent ("orchestrator") run** and runs **one child run per stage** (so each
stage's parameters are auto-logged and a non-zero exit aborts the pipeline). The pipeline YAML is logged as an
artifact of the parent run.

## Files

| Path | Role |
|------|------|
| `run_project.py` | Entrypoint: opens the MLflow experiment, launches the orchestrator |
| `src/stages/run.py` | Sequential orchestrator (`execute_stage` + the `skip` control key) |
| `src/stages/setup.py` | Stage 1 — make output/working dirs |
| `src/stages/compute_basins.py` | Stage 2 — LDD basins/tiles + clone/landmask maps (optional) |
| `src/stages/create_ini.py` | Stage 3 — render template → concrete `.ini` at a deterministic path |
| `src/stages/run_model.py` | Stage 4 — launch the model (parallel/serial, configurable) |
| `src/utils/io/parse_config.py` | YAML loader with `{{$ENV_VAR}}` expansion |
| `src/utils/shell.py` | `run_command` / `python_tool` subprocess helpers |
| `config/pipeline/pcrglobwb_pipeline.yaml` | The pipeline definition (edit this) |
| `config/pipeline/submit_pipeline.lsf` | LSF submission template |
| `conda_env/pcrglobwb_pipeline.yml` | pcraster env + `mlflow`, `fire`, `pyyaml` |

## Setup

```bash
# create the combined env (pcraster + harness deps) — keeps the name `pcrglobwb_python3`
conda env create -f conda_env/pcrglobwb_pipeline.yml
# ...or extend your existing env in place:
conda install -n pcrglobwb_python3 -c conda-forge pyyaml
conda run    -n pcrglobwb_python3 pip install "mlflow>=2.12,<3.13" fire
conda activate pcrglobwb_python3
```

## Configure

Edit `config/pipeline/pcrglobwb_pipeline.yaml`. Cluster paths come from two env vars (set in the LSF script),
expanded via `{{$VAR}}`:

- `PCRG_RUN_DIR` — run root (outputs, INIs, clone maps, basins land here)
- `PCRG_INPUT_DIR` — base of the PCR-GLOBWB input tree

Key conventions:

- **Optional stage 2** — set `skip: true` on `compute_basins` to reuse clone/landmask maps you already have.
- **The hand-off** — `create_ini`'s `output-path` is the exact `.ini` that `run_model`'s `config` consumes; set
  both to the same path (already wired in the template).
- **Parallel runs** — `clone-map`/`landmask` must contain `%s` (e.g. `.../clonemap_%s.map`); the runner
  substitutes the clone code (`M01`…). `n-tiles` should match the runner's expected clone codes (default 53).
- **Serial runs** — set `run_model.mode: serial` (drives `model/deterministic_runner.py`); `compute_basins`
  can be skipped.

## Run

**On the cluster (one LSF job):** edit the `#BSUB` directives and the two `export` lines in
`config/pipeline/submit_pipeline.lsf`, then:

```bash
bsub < config/pipeline/submit_pipeline.lsf
```

A parallel run launches all tiles on one node, so request at least `n_clones + 1` cores (54 for the default
53 clones + merging).

**Locally (smoke test, no model):** point the two env vars at scratch dirs, set `compute_basins.skip: true`,
`create_ini.novalidation: true`, and either `run_model.skip: true` or a tiny serial run:

```bash
export PCRG_RUN_DIR=/tmp/pcrg_run PCRG_INPUT_DIR=/tmp/pcrg_inputs
python run_project.py --config_file=config/pipeline/pcrglobwb_pipeline.yaml --experiment_name=smoke
```

## Track

```bash
mlflow ui            # default ./mlruns file store  →  http://127.0.0.1:5000
# or, for a sqlite store:
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each pipeline run shows the parent run (with the pipeline YAML artifact + your `tags`) and the four child
stage runs with their parameters. Use `--tracking_uri` on `run_project.py` (or `MLFLOW_TRACKING_URI`) to log
to a shared sqlite/remote store instead. `mlflow` is pinned `<3.13` so the default file store keeps working.
