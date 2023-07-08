# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2 (TF2)


Implementation of *Model-Agnostic Meta-Learning (MAML)* applied on Reinforcement Learning problems in TensorFlow 2.

## Usage
You can use the [`main.py`](main.py) script in order to train the algorithm with MAML.
```
python main.py --env-name 2DNavigation-v0 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 500 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15
```
 
Scripts were tested with Python 3.7.

## References

```