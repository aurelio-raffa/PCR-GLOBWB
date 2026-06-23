# MLflow-tracked PCR-GLOBWB pipeline

A thin orchestration layer (ported from [plumber](https://github.com/aurelio-raffa/plumber)) that runs a
whole PCR-GLOBWB experiment as **one MLflow-tracked pipeline**, submittable as a single LSF job:

1. **setup** — create the output / working directories
2. **compute_basins** *(optional)* — compute LDD basins/tiles and the per-tile clone & landmask maps
3. **inspect_partition** *(optional)* — validate inputs + tile summary + partition image (runnable standalone on a partition NPZ or a directory of clone/landmask maps)
4. **create_ini** — instantiate a concrete `.ini` from a template
5. **run_model** — launch PCR-GLOBWB (parallel/tiled or serial)
6. **plot_output** *(optional, final)* — plot a model output variable (map / animation / time series)

Each step is a Fire-CLI under `src/stages/`. The decomposition and INI logic lives in `src/utils/` as plain
functions; the stages and the root CLI tools both call those functions, so there is **one implementation**
shared between the pipeline and the standalone command-line tools (no shelling out).

## How it runs

```
run_project.py  ──mlflow.projects.run──▶  src/stages/run.py  (orchestrator: 1 parent MLflow run)
                                              │   for each stage:  mlflow.run ─▶ python src/stages/<stage>.py --k v
                                              ├─ setup.py            → os.makedirs(...)
                                              ├─ compute_basins.py   → src.utils.ldd_basins.compute_ldd_basins
                                              │                        + src.utils.tile_clone_maps.create_tile_clone_maps
                                              ├─ inspect_partition.py→ src.utils.partition_summary.inspect_partition
                                              ├─ create_ini.py       → src.utils.ini_config.create_ini_config  (→ output-path)
                                              ├─ run_model.py        → subprocess: model/parallel_pcrglobwb_runner*.py
                                              │                                    | model/deterministic_runner.py
                                              └─ plot_output.py      → src.utils.plot_output.plot_output
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
| `src/stages/inspect_partition.py` | Stage 3 *(optional)* — validate + tile summary + partition image (also standalone) |
| `src/stages/create_ini.py` | Stage 4 — Fire wrapper over `create_ini_config` (writes a deterministic `.ini`) |
| `src/stages/run_model.py` | Stage 5 — launch the model (parallel/serial, configurable) |
| `src/stages/plot_output.py` | Stage 6 *(optional, final)* — plot a model output variable |
| `src/utils/ldd_basins.py` | LDD decomposition implementation (`compute_ldd_basins`) |
| `src/utils/tile_clone_maps.py` | Per-tile clone/landmask implementation (`create_tile_clone_maps`) |
| `src/utils/ini_config.py` | INI templating implementation (`create_ini_config`) |
| `src/utils/partition_summary.py` | Validate + tile summary + partition image (`inspect_partition`: NPZ / extents / maps dir) |
| `src/utils/plot_output.py` | netCDF output plotter (`plot_output`) |
| `src/utils/plotting.py` | **Shared** matplotlib helpers (used by both plotting stages) |
| `src/utils/io/parse_config.py` | YAML loader with `{{$ENV_VAR}}` expansion |
| `src/utils/shell.py` | `run_command` / `python_tool` (used by the `run_model` stage) |
| `compute_ldd_basins.py`, `create_tile_clone_maps.py`, `create_ini_config.py`, `inspect_partition.py`, `plot_simulation_output.py` | Root **argparse CLI shims** over the `src/utils` functions |
| `plot_simulation_output_notebook.py` | Self-contained Jupyter `quicklook()` helper (numpy/netCDF4/matplotlib only) |
| `create_job_file.py` | Unified LSF generator: `deterministic` / `parallel` / `pipeline` modes |
| `create_batch_job_file.py`, `create_parallel_batch_job_file.py` | Deprecation shims → `create_job_file.py` |
| `config/pipeline/pcrglobwb_pipeline.yaml` | The pipeline definition (edit this) |
| `conda_env/pcrglobwb_pipeline.yml` | pcraster env + `mlflow`, `fire`, `pyyaml`, `matplotlib` |

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

Edit `config/pipeline/pcrglobwb_pipeline.yaml`. Every path/pattern in it is an environment-variable
placeholder `{{$VAR}}`, expanded from the process environment when the YAML is loaded (an undefined var →
`''`). The values are injected at submission time by `create_job_file.py pipeline` (see **Run**), so **no real
path is ever committed** — the repository only ever contains the placeholders, never your cluster paths.

### Placeholders

| Placeholder | Meaning | Consumed by (YAML key) |
|-------------|---------|------------------------|
| `PCRG_RUN_DIR` | run root — `output/`, `ini/`, `reports/`, `clone_maps/`, `basins/` land here | `setup.*`, `create_ini.output-dir` / `output-path`, `run_model.config`, `inspect_partition.output-*`, `plot_output.*` |
| `PCRG_INPUT_DIR` | base of the PCR-GLOBWB input tree | `create_ini.input-dir` |
| `CLONE_MAP_DIR` | directory holding the per-tile clone & landmask maps | `inspect_partition.maps-dir`; `create_ini.clone-map` / `landmask` (as `{{$CLONE_MAP_DIR}}/{{$…_PATTERN}}`) |
| `CLONE_MAP_PATTERN` | clone-map filename pattern — **must contain `%s`** (e.g. `clone_%s.map`) | `inspect_partition.clone-pattern`, `create_ini.clone-map` |
| `LANDMASK_PATTERN` | landmask filename pattern — **must contain `%s`** (e.g. `mask_%s.map` or `landmask_%s.map`) | `inspect_partition.landmask-pattern`, `create_ini.landmask` |

Add your own placeholders freely: any `{{$FOO}}` in the YAML is filled from `$FOO` (set it via
`create_job_file.py pipeline --env FOO=...`, see **Run**).

### Wiring notes

- **Provided clone/landmask maps (the shipped default).** The template points at an *existing* set of maps
  rather than generating tiles: `inspect_partition` runs in **directory mode** (`maps-dir` +
  `clone-pattern` + `landmask-pattern`) to validate them and write a summary + image to `reports/`, and
  `create_ini` assembles the INI `cloneMap` / `landmask` fields as `{{$CLONE_MAP_DIR}}/{{$CLONE_MAP_PATTERN}}`
  and `{{$CLONE_MAP_DIR}}/{{$LANDMASK_PATTERN}}` (both keep the `%s`, which the parallel runner substitutes per
  clone). For the official 05-arcmin masks, e.g. `CLONE_MAP_DIR=path/to/clone_landmask_maps`,
  `CLONE_MAP_PATTERN=clone_%s.map`, `LANDMASK_PATTERN=mask_%s.map`.
- **Generate tiles instead.** Add the optional `compute_basins` stage (it writes `clonemap_%s.map` /
  `landmask_%s.map` into `{{$PCRG_RUN_DIR}}/clone_maps`) and set the patterns/`CLONE_MAP_DIR` to point there.
- **The hand-off** — `create_ini.output-path` is the exact `.ini` that `run_model.config` consumes (same path).
- **Reports** — `setup.reporting_dir` (`{{$PCRG_RUN_DIR}}/reports`) is where `inspect_partition` and
  `plot_output` write their summary / figures.
- **Parallel vs serial** — `clone-map` / `landmask` must contain `%s` and `clone-areas` must match the runner's
  clone codes; for a single global/serial run set `run_model.mode: serial`.

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
    --repo_dir  "$PWD" \
    --env CLONE_MAP_DIR=path/to/clone_landmask_maps \
    --env "CLONE_MAP_PATTERN=clone_%s.map" \
    --env "LANDMASK_PATTERN=mask_%s.map"
# prints e.g. pipeline_2606231530.lsf ; then:
bsub < pipeline_2606231530.lsf
```

`--run_dir` / `--input_dir` are shorthand for `PCRG_RUN_DIR` / `PCRG_INPUT_DIR`; every other placeholder is set
with a repeatable `--env KEY=VALUE` (quote values containing `%s`). Each one becomes an `export KEY=VALUE` line
in the generated (git-ignored) `.lsf`, so your real paths live only in that file, never in the repo.

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
export CLONE_MAP_DIR=path/to/clone_landmask_maps CLONE_MAP_PATTERN='clone_%s.map' LANDMASK_PATTERN='mask_%s.map'
python run_project.py --config_file=config/pipeline/pcrglobwb_pipeline.yaml --experiment_name=smoke
```

## Track

```bash
mlflow ui            # default ./mlruns file store  →  http://127.0.0.1:5000
# or, for a sqlite store:
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run shows the parent run (pipeline-YAML artifact + your `tags`) and one child run per stage with their
parameters. Use `--tracking_uri` on `run_project.py` (or `create_job_file.py pipeline --tracking_uri`, or
`MLFLOW_TRACKING_URI`) to log to a shared sqlite/remote store. `mlflow` is pinned `<3.13` so the default file
store keeps working.
