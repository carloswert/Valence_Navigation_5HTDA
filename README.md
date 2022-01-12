## Code for Dopamine and serotonin interplay for valence-based spatial learning (Wert-Carvajal _et al.,_ 2022)

Code for a valence-based RL model of navigation in the hippocampus through DA/5-HT balance.
authors: [Carlos Wert-Carvajal](carloswertcarvajal@gmail.com), [Melissa Reneaux](reneauxm5@gmail.com), [Tatjana Tchumatchenko](mailto:tatjana.tchumatchenko@uni-mainz.de) and [Claudia Clopath](c.clopath@imperial.ac.uk)

The repository contains:

1. Code for simulations: Our implementation is based on [Zannone _et al.,_ (2017)](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=245018). We include Python codes for STDP and rate-based versions of sequential weight change (SWC) and competitive weight change (CWC). The output is a pickle file to be analysed.
2. Code for statistical analysis: Jupyter notebook
3. Files with the datasets used in the article: 

## Setup
Simulations require Python 3.7.9 and:

- Numba 0.50.1
- Numpy 1.18.5
- Matplotlib 3.2.2
- Scipy 1.4.1

Other requirements can be found in the requirements.txt file


## Run simulations

Codes for SWC are named as seq_neuromod and those for CWC are comp_neuromod. Simulations with more than one trial (-t) are runned in parallel, which can be optimised by changing the Pool() option of Python's multiprocessing library.

For execution:
```

python seq_neuromod_MWM.py -o <name_job> -e <episodes> -t <trials> -s

```

We also provide a PBS-compatible code for HPC systems that is able to schedule jobs according to the resources available.

## Data Analysis

The main figures can be recreated with the notebook file Data_Analysis.ipynb and the files contained in the Data_Analysis folder.
