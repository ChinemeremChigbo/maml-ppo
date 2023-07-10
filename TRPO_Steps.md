# Running Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2 (TF2) from scratch

This tutorial assumes a completely fresh installation of Ubuntu. To just run MAML TRPO start at step 3

## 1. Clone the repo:

## Use apt to install git
```bash
sudo apt install git
```

## Clone the relevant git repo
```bash
git clone https://github.com/schneimo/maml-rl-tf2.git
```

## Navigate into the git repo folder
```bash
cd maml-rl-tf2/
```

## 2. Get Python 3.7.17

## Install Curl
```bash
sudo apt install curl
```

## Install pyenv

### Curl pyenv from using bash
```bash
curl https://pyenv.run | bash
```

### Update ~/.bashrc with the relevant lines
```bash
printf 'export PATH="$HOME/.pyenv/bin:$PATH"\neval "$(pyenv init -)"\neval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

### Reload the shell
```bash
exec "$SHELL"
```

## Install Python Requirements
```bash
sudo apt-get install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## Install Python 3.7.17
```bash
sudo pyenv install 3.7.17
```

## 3. Create a virtual environment

## Use Python 3.7.17
```bash
pyenv local 3.7.17
```

## Create the virtual environment
```bash
python3 -m venv env3.7
```

## Activate virtual environment
```bash
source env3.7/bin/activate
```

## Edit the requirements.txt file
```bash
printf 'garage==2021.3.0\nmujoco_py==2.0.2.8\ngym==0.17.2' > requirements.txt'
```

## Install the given requirements
```bash
python3 -m pip install -r requirements.txt
```

## Run the main.py file
```bash
python3 main.py --env-name 2DNavigation-v0 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 500 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15
```