# MLflow-tracked PCR-GLOBWB pipeline

A thin orchestration layer (ported from [plumber](https://github.com/aurelio-raffa/plumber)) that runs a
whole PCR-GLOBWB experiment as **one MLflow-tracked pipeline**, submittable as a single LSF job:

1. **setup** — create the output / working directories
2. **compute_basins** *(optional)* — compute LDD basins/tiles and the per-tile clone & landmask maps
3. **create_ini** — instantiate a concrete `.ini` from a template
4. **run_model** — launch PCR-GLOBWB (parallel/tiled or serial)

Each step is a Fire-CLI under `src/stages/`. The decomposition and INI logic lives in `src/utils/` as plain
functions; the stages and the root CLI tools both call those functions, so there is **one implementation**
shared between the pipeline and the standalone command-line tools (no shelling out).

## How it runs

```
run_project.py  ──mlflow.projects.run──▶  src/stages/run.py  (orchestrator: 1 parent MLflow run)
                                              │   for each stage:  mlflow.run ─▶ python src/stages/<stage>.py --k v
                                              ├─ setup.py          → os.makedirs(...)
                                              ├─ compute_basins.py → src.utils.ldd_basins.compute_ldd_basins
                                              │                      + src.utils.tile_clone_maps.create_tile_clone_maps
                                              ├─ create_ini.py     → src.utils.ini_config.create_ini_config  (→ output-path)
                                              └─ run_model.py      → subprocess: model/parallel_pcrglobwb_runner*.py
                                                                                 | model/deterministic_runner.py
```

The orchestrator opens **one parent ("orchestrator") run** and runs **one child run per stage** (so each
stage's parameters are auto-logged and a non-zero exit aborts the pipeline). The pipeline YAML is logged as an
artifact of the parent run. `skip: true` on a stage makes it optional (used for `compute_basins`).

## Files

| Path | Role |
|------|------|
| `run_project.py` | Entrypoint: opens the MLflow experiment, launches the orchestrator |
| `src/stages/run.py` | Sequential orchestrator (`execute_stage` + the `skip` control key) |
| `src/stages/setup.py` | Stage 1 — make output/working dirs |
| `src/stages/compute_basins.py` | Stage 2 — Fire wrapper over the two `src/utils` functions (optional) |
| `src/stages/create_ini.py` | Stage 3 — Fire wrapper over `create_ini_config` (writes a deterministic `.ini`) |
| `src/stages/run_model.py` | Stage 4 — launch the model (parallel/serial, configurable) |
| `src/utils/ldd_basins.py` | LDD decomposition implementation (`compute_ldd_basins`) |
| `src/utils/tile_clone_maps.py` | Per-tile clone/landmask implementation (`create_tile_clone_maps`) |
| `src/utils/ini_config.py` | INI templating implementation (`create_ini_config`) |
| `src/utils/io/parse_config.py` | YAML loader with `{{$ENV_VAR}}` expansion |
| `src/utils/shell.py` | `run_command` / `python_tool` (used by the `run_model` stage) |
| `compute_ldd_basins.py`, `create_tile_clone_maps.py`, `create_ini_config.py` | Root **argparse CLI shims** over the `src/utils` functions (original CLIs preserved) |
| `create_job_file.py` | Unified LSF generator: `deterministic` / `parallel` / `pipeline` modes |
| `create_batch_job_file.py`, `create_parallel_batch_job_file.py` | Deprecation shims → `create_job_file.py` |
| `config/pipeline/pcrglobwb_pipeline.yaml` | The pipeline definition (edit this) |
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

Edit `config/pipeline/pcrglobwb_pipeline.yaml`. Cluster paths come from env vars expanded via `{{$VAR}}`
(set by `create_job_file.py pipeline` at submission time, so **no real path is ever committed**):

- `PCRG_RUN_DIR` — run root (outputs, INIs, clone maps, basins land here)
- `PCRG_INPUT_DIR` — base of the PCR-GLOBWB input tree

Key conventions:

- **Optional stage 2** — set `skip: true` on `compute_basins` to reuse clone/landmask maps you already have.
- **The hand-off** — `create_ini`'s `output-path` is the exact `.ini` that `run_model`'s `config` consumes
  (already wired to the same path in the template).
- **Parallel runs** — `clone-map`/`landmask` must contain `%s` (e.g. `.../clonemap_%s.map`). `n-tiles` should
  match the runner's expected clone codes (default 53).
- **Serial runs** — set `run_model.mode: serial`; `compute_basins` can be skipped.

## Run (one LSF job)

Generate the submission file **on the HPC** with `create_job_file.py pipeline`, passing the real paths as
arguments (they only ever land in the generated `.lsf`, which is git-ignored):

```bash
python create_job_file.py pipeline \
    --nc 54 --mem 128G --jq <QUEUE> --wd <WORKDIR> --pc <PROJECT_CODE> \
    --conda_env pcrglobwb_python3 --excl \
    --experiment_name pcrglobwb_run01 \
    --run_dir   /scratch/$USER/pcrglobwb/run01 \
    --input_dir /data/pcrglobwb/inputs \
    --repo_dir  "$PWD"
# prints e.g. pipeline_2606231530.lsf ; then:
bsub < pipeline_2606231530.lsf
```

A parallel run launches all tiles on one node, so request at least `n_clones + 1` cores (54 for the default
53 clones + merging). The same tool still generates the classic single-runner jobs:

```bash
python create_job_file.py deterministic --nc 8  --mem 32G  ... --config <ini>     # serial runner
python create_job_file.py parallel      --nc 54 --mem 128G ... --config <ini> --excl  # parallel runner
```

**Locally (smoke test, no model):** point the env vars at scratch dirs, set `compute_basins.skip: true`,
`create_ini.novalidation: true`, and `run_model.skip: true`:

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

Each run shows the parent run (pipeline-YAML artifact + your `tags`) and the four child stage runs with their
parameters. Use `--tracking_uri` on `run_project.py` (or `create_job_file.py pipeline --tracking_uri`, or
`MLFLOW_TRACKING_URI`) to log to a shared sqlite/remote store. `mlflow` is pinned `<3.13` so the default file
store keeps working.
