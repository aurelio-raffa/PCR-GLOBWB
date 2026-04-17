# PCR-GLOBWB

PCR-GLOBWB (PCRaster Global Water Balance) is a large-scale hydrological model intended for global to regional studies and developed at the Department of Physical Geography, Utrecht University (Netherlands). This repository holds the model scripts of PCR-GLOBWB.
Main reference/paper: Sutanudjaja, E. H., van Beek, R., Wanders, N., Wada, Y., Bosmans, J. H. C., Drost, N., van der Ent, R. J., de Graaf, I. E. M., Hoch, J. M., de Jong, K., Karssenberg, D., López López, P., Peßenteiner, S., Schmitz, O., Straatsma, M. W., Vannametee, E., Wisser, D., and Bierkens, M. F. P.: PCR-GLOBWB 2: a 5 arcmin global hydrological and water resources model, Geosci. Model Dev., 11, 2429-2453, https://doi.org/10.5194/gmd-11-2429-2018, 2018.

This repository was forked to allow minor customization, especially to run the model on Linux clusters via the 
[IBM Spectrum LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0) job scheduling system.
The main contributions in this regard are explained in the ["Scheduling Runs with IBM Spectrum LSF"](#scheduling-runs-with-ibm-spectrum-lsf)
and the ["Configuration File Recipes"](#configuration-file-recipes) Sections below.


## Input and output files (including OPeNDAP-based access: https://opendap.4tu.nl/thredds/catalog/data2/pcrglobwb/catalog.html)

The input files for the runs made in the aformentioned paper (Sutanudjaja et al., 2018) are available on the OPeNDAP server: https://opendap.4tu.nl/thredds/catalog/data2/pcrglobwb/version_2019_11_beta/pcrglobwb2_input/catalog.html. The OPeNDAP protocol (https://www.opendap.org) allow users to access PCR-GLOBWB input files from the remote server and perform PCR-GLOBWB runs **without** the need to download the input files (with total size ~250 GB for the global extent).

Some output files are also provided: https://opendap.4tu.nl/thredds/catalog/data2/pcrglobwb/version_2019_11_beta/example_output/global_05min_gmd_paper_output/catalog.html. More output files are available on https://geo.data.uu.nl/research-pcrglobwb/pcr-globwb_gmd_paper_sutanudjaja_et_al_2018/ (for requesting access, please send an e-mail to E.H.Sutanudjaja@uu.nl).


## How to install

Please follow the following steps required to install PCR-GLOBWB:

 1. You will need a working Python environment, we recommend to install Miniconda, particularly for Python 3. Follow their instructions given at https://docs.conda.io/en/latest/miniconda.html. The user guide and short reference on conda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).

 2. Get the requirement or environment file from this repository [conda_env/pcrglobwb_py3.yml](conda_env/pcrglobwb_py3.yml) and use it to install all modules required (e.g. PCRaster, netCDF4) to run PCR-GLOBWB:

    `conda env create --name pcrglobwb_python3 -f pcrglobwb_py3.yml`

    This will create a environment named *pcrglobwb_python3*.

 3. Activate the environment in a command prompt:

    `conda activate pcrglobwb_python3`

 4. Clone or download this repository. We suggest to use the latest version of the model, which should also be in the default branch. 

    `git clone https://github.com/UU-Hydro/PCR-GLOBWB_model.git`

    This will clone PCR-GLOBWB into the current working directory.


## PCR-GLOBWB configuration .ini file

For running PCR-GLOBWB, a configuration .ini file is required. Some configuration .ini file examples are given in the *config* directory. To be able to run PCR-GLOBWB using these .ini file examples, there are at least two things that must be adjusted. 

First, please make sure that you edit or set the *outputDir* (output directory) to the directory that you have access. You do not need to create this directory manually.  

Moreover, please also make sure that the *cloneMap* file is stored locally in your computing machine. The *cloneMap* file defines the spatial resolution and extent of your study area and must be in the pcraster format. Some examples are given in this repository [clone_landmask_maps/clone_landmask_examples.zip](clone_landmask_maps/clone_landmask_examples.zip).

By default, the configuration .ini file examples given in the *config* directory will use PCR-GLOBWB input files from the 4TU.ResearchData server, as set in their *inputDir* (input directory). 

`inputDir = https://opendap.4tu.nl/thredds/dodsC/data2/pcrglobwb/version_2019_11_beta/pcrglobwb2_input/`

This can be adjusted to any (local) locations, e.g. if you have the input files stored locally in your computing machine. 


> **Beware**
> 
> The `.ini` files provided in the original repository contain some error with respect to the [provided input files](#input-and-output-files-including-opendap-based-access-httpsopendap4tunlthreddscatalogdata2pcrglobwbcataloghtml), mainly:
> 1. wrong file extension (i.e., `.nc` instead of `.map`)
> 2. nonexistent paths (e.g., `/initialConditions/non-natural/consistent_run_201903XX/1999/`)
> 
> Reports on the errors that were detected in the files [`setup_05min.ini`](config%2Fsetup_05min.ini) and [`setup_30min.ini`](config%2Fsetup_30min.ini),
> generated by using the utility [`fix_files_mapping.py`](fix_files_mapping.py), can be found in [`05_min_corrections.txt`](05_min_corrections.txt) and [`30_min_corrections.txt`](30_min_corrections.txt)

### Configuration File Recipes

To address the mapping issues mentioned in the [previous section](#pcr-globwb-configuration-ini-file)
and preserve anonymity of users' operating environments while maintaining reproducibility (i.e., writing no explicit
paths in config. files in a publicly available repository), a [`create_ini_config.py`](create_ini_config.py) 
utility has been made available to create `.ini` files on-the-fly from command line.

The script generates a new configuration (.ini) file from a template by replacing placeholders with values provided 
via command-line arguments. It adds a timestamp and metadata to the file and saves it with a uniquely generated 
name based on the current date, time, and user-defined identifier.

An example of usage pattern is the following:
```shell
python create_ini_config.py \
--name=NAME_FOR_THE_EXPERIMENT \
--institution=YOUR_INSITUTION \
--title=TITLE_FOR_THE_RUN \
--base_ini=config/30min.ini \
--outputDir=YOUR_OUTPUT_DIR \
--cloneMap=./clone_landmask_maps/clone_global_30min.map \
--inputDir=YOUR_DATA_DIR
```

More information can be printed out on the terminal by executing:
```shell
python create_ini_config.py --help
```

## How to run

### Python Code Execution

Please make sure that the correct conda environment in a command prompt:

`conda activate pcrglobwb_python3`

Go to to the PCR-GLOBWB *model* directory. You can start a PCR-GLOBWB run using the following command: 

`python deterministic_runner.py <ini_configuration_file>`

where <ini_configuration_file> is the configuration file of PCR-GLOBWB. 

> **Note**
> 
> This way of running the model is not ideal if you are on a cluster with several users.
> Please refer to the [next section](#scheduling-runs-with-ibm-spectrum-lsf) for more info on
> Job execution on clusters implementing IBM Spectrum LSF

### Scheduling Runs with IBM Spectrum LSF

A utility to automate `.lsf` job submission file generation for [IBM Spectrum LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0)
is provided in [`create_batch_job_file.py`](create_batch_job_file.py).

The benefit is that no explicit LSF file containing sensitive info has to be versioned, but creation on-the-fly is
possible. An example usage is the following:

```shell
python create_batch_job_file.py \
--nc=72 \
--mem=128G \
--jq=YOUR_QUEUE \
--wd=PATH/TO/WORKING/DIR \
--pc=PRJC0D3 \
--conda_env=pcrglobwb_python3 \
--config=PATH/TO/CONFIG/INI \
--excl
```

The utility prints on the terminal the name of the generated `.lsf` file,
which follows a naming pattern of the type `job_{%y%m%d%H%M}.lsf`.

More information can be printed out on the terminal by executing:
```shell
python create_batch_job_file.py --help
```

## Exercises/cooking recipes 

We included some exercise/cooking recipes for running PCR-GLOBWB. You can find these documents in the folder [exercise](exercise) within this repository. While these exercises were generally designed for our own computing facilities (e.g. velocity and eejit servers), they should be adaptable for use on other computing machines.
