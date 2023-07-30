# Running Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) TRPO and PPO

This tutorial assumes a completely fresh installation of Ubuntu. If dependencies
are already installed then simply run the "Activate Virtual Environment",
"Install the Given Requirements", and "Run the Relevant `Python` File" steps.
This repo is inspired by Moritz Schneider's implementation of MAML TRPO
[schneimo/maml-rl-tf2](https://github.com/schneimo/maml-rl-tf2/) (TensorFlow) as
well as the rlworkgroup's implementation of MAML PPO
[rlworkgroup/garage](https://github.com/rlworkgroup/garage) (PyTorch /
TensorFlow).

## Clone the Repo

### Update `apt` Packages

```bash
sudo apt update
```

### Use `apt` to Install `git`

```bash
sudo apt install git
```

### Create a Personal Access Token on GitHub

From your GitHub account, go to Settings → Developer Settings → Personal Access
Tokens → Tokens (Classic) → Generate New Token → Generate New Token (Classic) → Add a relevant "Note" → Select Scope of "Repo" → Fill out the Remainder of the Form  → Generate Token → Copy the Generated Token, it will be something like
`ghp_randomly_generated_personal_access_token`

### Git Clone Using your Personal Access Token

```bash
git clone https://ghp_randomly_generated_personal_access_token@github.com/ChinemeremChigbo/maml-ppo.git
```

### Navigate into the `git` Repo Folder

```bash
cd maml-ppo/
```

## Get `Python 3.7.16`

### Install `Python` Requirements

```bash
sudo apt install curl build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Curl `pyenv` from Using `bash`

```bash
curl https://pyenv.run | bash
```

### Update `~/.bashrc` with the Relevant Lines

```bash
printf "%s\n" '' 'export PATH="$HOME/.pyenv/bin:$PATH"' 'eval "$(pyenv init -)"' 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

### Reload the `~/.bashrc`

```bash
source ~/.bashrc
```

### Install `Python 3.7.16`

```bash
pyenv install 3.7.16
```

## Get `mujoco150`

### Add universe `apt`
```bash
sudo add-apt-repository universe && sudo apt upgrade
```

### Install `mujoco` Requirements

```bash
sudo apt install ffmpeg patchelf unzip libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Download `mujoco150`

```bash
wget https://www.roboti.us/download/mjpro150_linux.zip
```

### Download the `mujoco` license

```bash
wget https://www.roboti.us/file/mjkey.txt
```

### Unzip the `mujoco150` zip folder

```bash
unzip mjpro150_linux.zip
```

### Remove the `mujoco150` zip folder

```bash
rm mjpro150_linux.zip
```

### Make a `mujoco` directory in the current user's folder

```bash
mkdir $HOME/.mujoco
```

### Move `mujoco150` to the mujoco folder

```bash
mv mjpro150 $HOME/.mujoco
```

### Move the `mujoco` License to the `mujoco` Folder

```bash
mv mjkey.txt $HOME/.mujoco
```

### Update `~/.bashrc` with the Relevant Lines

```bash
printf "%s\n" '' 'export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin' 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.bashrc
```

### Reload the `~/.bashrc`

```bash
source ~/.bashrc
```

## Create a Virtual Environment

### Use `Python 3.7.16`

```bash
pyenv local 3.7.16
```

### Make the Virtual Environment Folder

```bash
python3 -m venv env3.7
```

### Activate Virtual Environment

```bash
source env3.7/bin/activate
```

### Install wheel

```bash
pip install wheel==0.40.0
```
## Run the Relevant `Python` File

### Install the Given Requirements

```bash
python3 -m pip install -r requirements.txt
```

### Run the `main_trpo.py` File *To run this, uncomment mujoco requirement from requirements.txt*

```bash
python3 main_trpo.py --env-name HalfCheetahDir-v1 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 5 --meta-batch-size 10 --num-layers 2 --hidden-size 100 --num-batches 1 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 10 --save-iters 1
```

### Run the `experiments.py` File

```bash
python3 experiments.py
```

### Run the `main_maml_ppo.py` File

```bash
python3 main_maml_ppo.py --epochs=1 --episodes_per_task=1
```

### Run the `main_cav_ppo.py` File

```bash
python3 main_cav_ppo.py --epochs=1 --episodes_per_task=1
```

### Run the `main_cav_maml_ppo.py` File

```bash
python3 main_cav_maml_ppo.py --epochs=1
```

### Run the test2CAV File

Note that you can replace 2 with whichever CAV test is required

```bash
python3 test_2CAV_BFoptimal_Kaige.py
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
