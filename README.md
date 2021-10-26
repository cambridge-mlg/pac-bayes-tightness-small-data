# How Tight Can PAC-Bayes be in the Small Data Regime?

This is the code to reproduce all experiments for the following paper:

```
@inproceedings{Foong:2021:How_Tight_Can_PAC-Bayes_Be,
    title = {How Tight Can {PAC}-{Bayes} Be in the Small Data Regime?},
    year = {2021},
    author = {Andrew Y. K. Foong and Wessel P. Bruinsma and David R. Burt and Richard E. Turner},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {35},
    eprint = {https://arxiv.org/abs/2106.03542},
}
```

Every experiment creates a folder in `_experiments`.
The names of the files in those folders should be self-explanatory.

## Installation
First, create and activate a virtual environment for Python 3.8.

```bash
virtualenv venv -p python3.8 
source venv/bin/activate
```

Then install an appropriate GPU-accelerated version of
[PyTorch](https://pytorch.org/get-started/locally/).

Finally, install the requirements for the project.

```bash
pip install -e . 
```

You should now be able to run the below commands.

## Generating Datasets
In order to generate the synthetic 1D datasets used, run these commands from inside `classification_1d`:

```bash
python gen_data.py --class_scheme balanced --num_context 30 --name 30-context --num_train_batches 5000 --num_test_batches 64
python gen_data.py --class_scheme balanced --num_context 60 --name 60-context --num_train_batches 5000 --num_test_batches 64
```

The generated datasets will be in `pacbayes/_data_caches`

## Theory Experiments
See Figure 2 in Section 3 and Appendix G.

```bash
python theory_experiments.py --setting det1-1
python theory_experiments.py --setting det1-2
python theory_experiments.py --setting det2-1
python theory_experiments.py --setting det2-1

python theory_experiments.py --setting stoch1
python theory_experiments.py --setting stoch2
python theory_experiments.py --setting stoch3

python theory_experiments.py --setting random --random-seed 1 --random-better-bound maurer
python theory_experiments.py --setting random --random-seed 6 --random-better-bound catoni
```

## GNP Classification Experiments
See Figure 3 and 4 in Section 4 and Appendices I and J.
The numbers from the graphs can be found in `eval_metrics_no_post_opt.txt`
(without post optimisation) `eval_metrics_post_opt.txt` (with post optimisation).

```bash
MODEL_NONDDP=maurer MODEL_DDP=maurer-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer MODEL_DDP=maurer-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=convex-nonseparable MODEL_DDP=convex-nonseparable-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=convex-nonseparable MODEL_DDP=convex-nonseparable-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-optimistic MODEL_DDP=maurer-optimistic-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-optimistic MODEL_DDP=maurer-optimistic-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-inv MODEL_DDP=maurer-inv-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-inv MODEL_DDP=maurer-inv-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-inv-optimistic MODEL_DDP=maurer-inv-optimistic-ddp NUM_CONTEXT=30 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-inv-optimistic MODEL_DDP=maurer-inv-optimistic-ddp NUM_CONTEXT=30 ./run_GNP_prop_68.sh

MODEL_NONDDP=maurer MODEL_DDP=maurer-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer MODEL_DDP=maurer-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=convex-nonseparable MODEL_DDP=convex-nonseparable-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=convex-nonseparable MODEL_DDP=convex-nonseparable-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-optimistic MODEL_DDP=maurer-optimistic-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-optimistic MODEL_DDP=maurer-optimistic-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-inv MODEL_DDP=maurer-inv-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-inv MODEL_DDP=maurer-inv-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
MODEL_NONDDP=maurer-inv-optimistic MODEL_DDP=maurer-inv-optimistic-ddp NUM_CONTEXT=60 ./run_GNP_prop_024.sh
MODEL_NONDDP=maurer-inv-optimistic MODEL_DDP=maurer-inv-optimistic-ddp NUM_CONTEXT=60 ./run_GNP_prop_68.sh
```

## MLP Classification Experiments
See Appendix J.
The numbers from the graphs can be found in `eval_metrics_no_post_opt.txt`
(without post optimisation) `eval_metrics_post_opt.txt` (with post optimisation).

```bash
MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=30 ./run_MLP.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=30 ./run_MLP.sh

MODEL_NONDDP=catoni MODEL_DDP=catoni-ddp NUM_CONTEXT=60 ./run_MLP.sh
MODEL_NONDDP=kl-val MODEL_DDP=kl-val NUM_CONTEXT=60 ./run_MLP.sh
```
