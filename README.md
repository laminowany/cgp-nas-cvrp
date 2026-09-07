# CGP-NAS for the Capacitated Vehicle Routing Problem

This repository contains the implementation developed as part of my master's thesis **"Evolving Graph Neural Network Architectures with Cartesian Genetic Programming for the Capacitated Vehicle Routing Problem"**.

The project investigates the use of **Cartesian Genetic Programming (CGP)** as a **Neural Architecture Search (NAS)** method for automatically evolving neural network encoder architectures for the **Capacitated Vehicle Routing Problem (CVRP)**.

The implementation builds upon the Attention Model introduced by Kool et al. in [*Attention, Learn to Solve Routing Problems!*](https://arxiv.org/abs/1803.08475). The original decoder and reinforcement learning training procedure are preserved, while the encoder can be represented and evolved using CGP.


## Dependencies

The minimum required Python version is Python 3.10. Python dependencies are specified in `pyproject.toml`.

The `graphviz` system package is also required. On Ubuntu:

```bash
sudo apt install graphviz
```

## Overview

The software is run from the command line. The main entry point is `src/run.py`, which supports architecture search, genome training, dataset generation, and model evaluation.

Run commands from the `src/` directory:

```bash
python run.py --mode MODE [OPTIONS]
```

Available modes:

- `cgp_search`: CGP-based architecture search.
- `random_search`: random architecture search.
- `genome_evaluation`: training and evaluation of a specified genome.
- `scoring`: evaluation of a trained model checkpoint on a test dataset.
- `generate_validation_data`: generation of a fixed CVRP dataset.

## Common Parameters

| Parameter | Description |
| --- | --- |
| `--mode` | Execution mode. |
| `--seed` | Random seed. The default is `1234`; `-1` generates a random seed. |
| `--graph_size` | Number of customers in the CVRP instance. |
| `--epoch_size` | Number of training instances generated per epoch. |
| `--n_epochs` | Number of training epochs. The default is `100`. |
| `--budget` | Number of architecture evaluations during architecture search. The default is `200`. |
| `--x_dim`, `--y_dim` | Number of columns and rows in the CGP grid. |
| `--validation_set_path` | Path to a fixed validation dataset. |
| `--test_set_path` | Path to a test dataset. |
| `--checkpoint_path` | Path to a model checkpoint containing trained weights. |
| `--run_name` | Optional name added to the generated experiment directory. |
| `--no_progress_bar` | Disable the progress bar during training. |
| `--no_save_model` | Prevent model weights from being saved. |
| `--start_from_transformer` | Use the Transformer as the initial parent. |

The training procedure uses the rollout baseline and eight attention heads. These values are fixed internally by the implementation.

## CGP Search

To reproduce the CGP search from Experiment I:

```bash
python run.py \
  --mode cgp_search --x_dim 15 --y_dim 5 \
  --n_epochs 10 --epoch_size 12800 --graph_size 10 \
  --no_progress_bar --no_save_model --seed -1 \
  --validation_set_path data/dataset_10CVRP_seed_3232.pt
```

This uses the default computational budget of 200. To initialize the search from the Transformer for Experiment II, add `--start_from_transformer`:

```bash
python run.py \
  --mode cgp_search --start_from_transformer --x_dim 8 --y_dim 5 \
  --n_epochs 10 --epoch_size 12800 --graph_size 10 \
  --no_progress_bar --no_save_model --seed -1 \
  --validation_set_path data/dataset_10CVRP_seed_3232.pt
```

The number of Transformer layers depends on `x_dim`; the program creates as many layers as fit within the specified grid.

## Random Search

To run random search from Experiment I:

```bash
python run.py \
  --mode random_search --x_dim 15 --y_dim 5 \
  --n_epochs 10 --epoch_size 12800 --graph_size 10 \
  --no_progress_bar --no_save_model --seed -1 \
  --validation_set_path data/dataset_10CVRP_seed_3232.pt
```

Architectures are generated randomly from the same architecture search space.

## Genome Training and Evaluation

This mode creates an encoder from the architecture provided through `--genome`. To train EVO-3 from Experiment II on CVRP100:

```bash
python run.py \
  --mode genome_evaluation --x_dim 8 --y_dim 5 --graph_size 100 \
  --n_epochs 100 --epoch_size 1280000 \
  --genome "[None, (5, 0), (1, 17), (7, 18), (5, 27), (7, 20), (1, 21), (5, 22), (7, 7), (4, 0), (3, 33, 1), (4, 2), (6, 19), (1, 4), (7, 5), (2, 38), (6, 15), (2, 0), (5, (1, 9)), (5, 26), (3, 3, 1), (1, 12), (3, 5, -1), (4, 30), (2, 7), (7, 0), (6, 25), (3, 34, -1), (2, 35), (1, 28), (2, 21), (7, 38), (7, 23), (2, 0), (7, 25), (1, 34), (5, 3), (4, 28), (4, 13), (2, 30), (7, 23), (5, 24)]" \
  --validation_set_path data/dataset_100CVRP_seed_3232.pt
```

The value passed to `--genome` must use a valid Python list representation.

## Evaluation of a Trained Model

A trained model can be evaluated on an existing test dataset using `scoring`. To evaluate EVO-3 from Experiment II on the CVRP100 test dataset:

```bash
python run.py \
  --checkpoint_path ../saved_weights/EVO-3/epoch-99.pt \
  --mode scoring --x_dim 8 --y_dim 5 \
  --genome "[None, (5, 0), (1, 17), (7, 18), (5, 27), (7, 20), (1, 21), (5, 22), (7, 7), (4, 0), (3, 33, 1), (4, 2), (6, 19), (1, 4), (7, 5), (2, 38), (6, 15), (2, 0), (5, (1, 9)), (5, 26), (3, 3, 1), (1, 12), (3, 5, -1), (4, 30), (2, 7), (7, 0), (6, 25), (3, 34, -1), (2, 35), (1, 28), (2, 21), (7, 38), (7, 23), (2, 0), (7, 25), (1, 34), (5, 3), (4, 28), (4, 13), (2, 30), (7, 23), (5, 24)]" \
  --test_set_path data/dataset_100CVRP_seed_2323.pt
```

The program reconstructs the encoder from the genome, loads model parameters from the checkpoint, and evaluates the model on the specified test set. The final routing score and number of active encoder parameters are printed after evaluation.

Test datasets may be stored as PyTorch files or `.pkl` files. Pickle datasets are converted internally to the representation expected by the CVRP implementation.

## Dataset Generation

Generate a fixed CVRP dataset with:

```bash
python run.py \
  --mode generate_validation_data \
  --graph_size 10 \
  --seed 2323
```

The dataset is stored in the experiment output directory with a name containing the problem size and random seed, for example:

```text
dataset_10CVRP_seed_2323.pt
```

The generated dataset can later be supplied with `--validation_set_path` or `--test_set_path`.

## Output Files

Each run creates a timestamped directory:

```text
outputs/run_YYYYMMDDTHHMMSS_NAME/
```

Architecture search modes also create:

```text
genomes_full/
genomes_active/
parents/
```

`genomes_full` contains visualizations of complete CGP genotypes, including inactive nodes. `genomes_active` contains visualizations of the corresponding active computational graphs. `parents` stores visualizations of architectures selected as parents during evolution.

The program also records candidate genomes, evaluation scores, evolutionary progress, and budget usage during the experiment.
