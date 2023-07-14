# Running Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2 (TF2) from Scratch

This tutorial assumes a completely fresh installation of Ubuntu. To just run,
start at "Create a Virtual environment" step.
This repo is inspired by Moritz Schneider's implementation of MAML TRPO
[schneimo/maml-rl-tf2](https://github.com/schneimo/maml-rl-tf2/) (TensorFlow) as
well as the rlworkgroup's implementation of MAML PPO
[rlworkgroup/garage](https://github.com/rlworkgroup/garage) (PyTorch /
TensorFlow).

## Clone the Repo

### Use `apt` to Install `git`
```bash
sudo apt install git
```

### Configure your Git username
```bash
$ git config --global user.name "your_github_username"
```

### Configure your Git email
```bash
$ git config --global user.email "your_github_email"
```

### Create a Personal Access Token on GitHub
From your GitHub account, go to Settings → Developer Settings → Personal Access
Tokens (classic) → Fill out the Form →  Generate Token → Copy the
generated Token, it will be something like 
`ghp_randomly_generated_personal_access_token`

### Git Clone Using Your Personal Access Token
```bash
git clone https://ghp_randomly_generated_personal_access_token@github.com/ChinemeremChigbo/maml-ppo.git
```

### Navigate into the `git` Repo Folder
```bash
cd maml-ppo/
```

### Reload the Shell
```bash
sudo apt-get update
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
## Download `mujoco150`

```bash
wget https://www.roboti.us/download/mjpro150_linux.zip
```

### Download the mujoco license
```bash
wget https://www.roboti.us/file/mjkey.txt
```

### Install mujoco prerequisites
```bash
sudo apt install unzip libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Unzip the `mujoco150` zip folder
```bash
unzip mjpro150_linux.zip
```

### Remove the `mujoco150` zip folder
```bash
rm mjpro150_linux.zip
```

### Make a mujoco directory in the current user's folder
**The following commands assume $USER is the current user**
```bash
mkdir /home/$USER/.mujoco
```

### Move `mujoco150` to the mujoco folder
```bash
mv mjpro150 /home/$USER/.mujoco
```

### Move the mujoco license to the mujoco folder
```bash
mv mjkey.txt /home/$USER/.mujoco
```

### Update `~/.bashrc` with the Relevant Lines
```bash
printf 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chinemerem/.mujoco/mjpro150/bin' >> ~/.bashrc
```

### Reload the Shell
```bash
exec "$SHELL"
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
## Edit the requirements
```bash
printf "%s\n" "garage==2021.3.0" "wheel==0.40.0" "gym[mujoco]==0.17.2" "pytest==6.1.2" "sacred==0.8.1" "tensorboard==2.4.0" "tensorflow==2.3.1" "tensorflow-estimator==2.3.0" "coverage==5.3" "scipy==1.7.3" "matplotlib==3.5.3" "pandas==1.3.5" "sympy==1.10.1" > requirements.txt
```

### Install the Given Requirements
```bash
python3 -m pip install -r requirements.txt
```

### Use Pure-Python Parsing for Protocol Buffers
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### Run the `main_trpo.py` File
```bash
python3 main_trpo.py --env-name 2DNavigation-v0 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 500 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15
```

### Run the `main_ppo.py` File
```bash
python3 main_ppo.py
```

### Run the test2CAV File
Note that you can replace 2 with whichever CAV test is required
```bash
python3 test_2CAV_BFoptimal_Kaige.py

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