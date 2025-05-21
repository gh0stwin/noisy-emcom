# Noisy Emergent Communication

Official implementation of [Implicit Repair with Reinforcement Learning in Emergent Communication](https://arxiv.org/abs/2502.12624), accepted at [AAMAS 2025](https://aamas2025.org) as a full paper.

## Installation

### Setup

```bash
conda create -n noisy_emcom python=3.10.16 -y
conda activate noisy_emcom
conda install -c conda-forge poetry -y
poetry self update
poetry install  # if it fails, run `poetry lock` before
conda deactivate
conda activate noisy_emcom
```

### Datasets

1. Download [ImageNet](https://cloud.hlt.inesc-id.pt/nextcloud/index.php/s/2Pgy8d2QcKMRFYY) and [CelebA](https://cloud.hlt.inesc-id.pt/nextcloud/index.php/s/2Pgy8d2QcKMRFYY) datasets.

2. Paste unzipped dataset folders inside `$HOME/tensorflow_datasets/`.

## Replicating Experiments

### Train

At the end of training the experiment will save the model at `.tmp/cidre_ckpts/<model-name>.pkl`. A message similar to `I0519 21:02:42.351042 140703576343424 checkpointer.py:180] Checkpoint saved at: ./.tmp/cidre_ckpts/agentsd3db1be0857cd466d4ff7a37de2bc0b0.pkl` will be outputted before the experiment ends.

#### LG (S)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
python main.py \
    --config=src/noisy_emcom/configs/lg_rlss_config.py:$DATA \
    --config.random_seed=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff )) \
    --config.batch_size=$BS \
```

#### LG (RL)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
python main.py \
    --config=src/noisy_emcom/configs/lg_rlrl_config.py:$DATA \
    --config.random_seed=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff )) \
    --config.batch_size=$BS \
```

#### NLG

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
python main.py \
    --config=src/noisy_emcom/configs/nlg_rlrl_config.py:${DATA}_${NOISE} \
    --config.random_seed=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff )) \
    --config.batch_size=$BS \
```

### Test

#### Channel (Section 3.2)

##### LG (S)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlss_config.py:$DATA \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### LG (RL)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlrl_config.py:$DATA \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### NLG

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/nlg_rlrl_config.py:${DATA}_${NOISE} \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

#### Message (Section 3.3)

##### LG (S)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlss_config.py:$DATA \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### LG (RL)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlrl_config.py:$DATA \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### NLG

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/nlg_rlrl_config.py:${DATA}_${NOISE} \
    --config.experiment_mode=testchannel \
    --config.random_seed=42 \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

#### Inputs (Section 3.4)

##### LG (S)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlss_config.py:$DATA \
    --config.experiment_mode=testinput \
    --config.random_seed=42 \
    --config.experiment_kwargs.config.game.kwargs.logit.coeff_noise=1 \
    --config.experiment_kwargs.config.game.kwargs.logit.has_noise=True \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### LG (RL)

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/lg_rlrl_config.py:$DATA \
    --config.experiment_mode=testinput \
    --config.random_seed=42 \
    --config.experiment_kwargs.config.game.kwargs.logit.coeff_noise=1 \
    --config.experiment_kwargs.config.game.kwargs.logit.has_noise=True \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

##### NLG

```bash
DATA="imagenet"  # or DATA="celeba_logits"
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training  # model saved at the end of training  # model saved at the end of training  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/nlg_rlrl_config.py:${DATA}_${NOISE} \
    --config.experiment_mode=testinput \
    --config.random_seed=42 \
    --config.experiment_kwargs.config.game.kwargs.logit.coeff_noise=1 \
    --config.experiment_kwargs.config.game.kwargs.logit.has_noise=True \
    --config.batch_size=$BS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH
```

### ETL (Appendix)

`celeba_logits`, `celeba_noimg`, and `celeba` all have the same `x` inputs (logits), but different auxiliary data:

- `celeba_logits`: no auxiliary `y` data;
- `celeba_noimg`: auxiliary `y` data contains the *image class*, *image attributes*, and *image landmarks*;
- `celeba`: auxiliary `y` data contains the original image pixels, and all auxiliary data present in `celeba_noimg`.

#### Train

```bash
# DATA="imagenet" can be used with `discriminationdist` and `classification` tasks
# DATA="celeba_logits" can be used with `discriminationdist` task
# DATA="celeba_noimg" can be used with `classification` and `attribute` tasks
# DATA="celeba" can be used with `reconstruction` task
DATA="imagenet"
TASK="discriminationdist"  # or TASK="attribute", TASK="classification", TASK="reconstruction"
STEPS="10000"  # or STEPS="30000" for `classification` task and `celeba_logits` data
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of training  # model saved at the end of training  # model saved at the end of training  # model saved at the end of training
python main.py \
    --config=src/noisy_emcom/configs/etl_config.py:${DATA}_${TASK}_${NOISE} \
    --config.random_seed=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff )) \
    --config.training_steps=$STEPS \
    --config.save_checkpoint_interval=$STEPS \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH \
    --config.batch_size=4096  # Add this only when running `discriminationdist` task
```

#### Test

```bash
# DATA="imagenet" can be used with `discriminationdist` and `classification` tasks
# DATA="celeba_logits" can be used with `discriminationdist` task
# DATA="celeba_noimg" can be used with `classification` and `attribute` tasks
# DATA="celeba" can be used with `reconstruction` task
DATA="imagenet"
TASK="discriminationdist"  # or TASK="attribute", TASK="classification", TASK="reconstruction"
BS="1024"  # or BS="16", BS="64", BS="256"
NOISE="0.5"  # or a value between (0.0, 1.0)
MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl"  # model saved at the end of (LG) training; <file-name> follows the convention 'agents<hash>'
ETL_MODEL_PATH=".tmp/cidre_ckpts/<file-name>.pkl" # model saved at the end of (ETL) training; <file-name> follows the convention 'etl<hash>'
python main.py \
    --config=src/noisy_emcom/configs/etl_config.py:${DATA}_${TASK}_${NOISE} \
    --config.experiment_mode="test" \
    --config.random_seed=42 \
    --config.experiment_kwargs.config.checkpointing.restore_path=$MODEL_PATH \
    --config.experiment_kwargs.config.checkpoint_experiment.restore_path=$ETL_MODEL_PATH \
    --config.batch_size=4096  # Add this only when running `discriminationdist` task
```
