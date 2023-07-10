# Running Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2 (TF2) from Scratch

This tutorial assumes a completely fresh installation of Ubuntu. To just run
MAML TRPO start at "Create a Virtual environment" step.

## Clone the Repo

### Use `apt` to Install `git`
```bash
sudo apt install git
```

### Clone the Relevant `git` Repo
```bash
git clone https://github.com/schneimo/maml-rl-tf2.git
```

### Navigate into the `git` Repo Folder
```bash
cd maml-rl-tf2/
```

## Get `Python 3.7.17`

### Install Curl
```bash
sudo apt install curl
```

### Curl `pyenv` from Using `bash`
```bash
curl https://pyenv.run | bash
```

### Update `~/.bashrc` with the Relevant Lines
```bash
printf 'export PATH="$HOME/.pyenv/bin:$PATH"\neval "$(pyenv init -)"\neval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

### Reload the Shell
```bash
exec "$SHELL"
```

### Install `Python` Requirements
```bash
sudo apt-get install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Install `Python 3.7.17`
```bash
sudo pyenv install 3.7.17
```

## Create a Virtual Environment

### Use `Python 3.7.17`
```bash
pyenv local 3.7.17
```

### Make the Virtual Environment Folder
```bash
python3 -m venv env3.7
```

### Activate Virtual Environment
```bash
source env3.7/bin/activate
```

### Edit the `requirements.txt` File
```bash
printf 'garage==2021.3.0\nmujoco_py==2.0.2.8\ngym==0.17.2' > requirements.txt'
```

### Install the Given Requirements
```bash
python3 -m pip install -r requirements.txt
```

### Run the `main.py` File
```bash
python3 main.py --env-name 2DNavigation-v0 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 500 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15
```

## References

```latex
@misc{garage,
 author = {The garage contributors},
 title = {Garage: A toolkit for reproducible reinforcement learning research},
 year = {2019},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/rlworkgroup/garage}},
 commit = {be070842071f736eb24f28e4b902a9f144f5c97b}
}
```

```latex
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```