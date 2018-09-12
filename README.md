# Expert-augmented actor-critic for Montezuma's Revenge

This repository contains code which combines the [ACKTR
algorithm](https://arxiv.org/abs/1708.05144) with expert trajctories.  Our
experiments led to be the best publicly available result for Montezuma's
Revenge, including a run which scored 804,900 points.

[Click here to see the Arxiv paper.](https://arxiv.org/pdf/1809.03447)

[Click here to see Montezuma's Revenge gameplay videos](https://www.youtube.com/playlist?list=PL1dkNgD0lNRZLL7STGpaw8hytlcnUw5ok)

[Click here to see videos of bug exploits in Montezuma's Revenge](https://www.youtube.com/playlist?list=PL1dkNgD0lNRaPCiSoe1T3V4xcro2yGJ2W)

In the article we evaluate our algorithm on two environments with sparse
rewards: Montezuma's Revenge and a maze from the ViZDoom suite. In the case of
Montezuma's Revenge, an agent trained with our method achieves very good
results, consistently scoring above 27,000 points (in many experiments beating
the [first world](https://atariage.com/2600/archives/strategy_MontezumasRevenge_Level1.html)).
With an appropriate choice of hyperparameters, our algorithm surpasses the
performance of the expert data.

Below you can see two excerpts from the gameplay.  On the right, the agent
exploits an unreported bug in Montezuma's Revenge and scores 804 900 points.
On the left, you can see part of typical evaluation in the [second
world](https://atariage.com/2600/archives/strategy_MontezumasRevenge_Level2.html).

![No bug gameplay](https://ghostfacekillah.github.io/img/expert/life.gif) ![bug gameplay](https://ghostfacekillah.github.io/img/expert/bug.gif)

Our algorithm is easy to understand and implement. Its core can be expressed in
one formula:

![loss](https://ghostfacekillah.github.io/img/expert/loss-expression.png)

On the left, you can see the standard actor-critic [A2C loss
function](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#a2c).
On the right, you can see the new loss term added by the algorithm, which is
similar in spirit to the former one, but expectations are computed over batches
of expert transitions sampled from a fixed dataset.  Below we include
pseudocode of our algorithm :

<img src="https://www.mimuw.edu.pl/~henrykm/eaacktr.png" width=400/>


This code is based on [OpenAI's baselines](https://github.com/openai/baselines).

### How to run the experiments?

The optimal hardware for running these experiments is:

* 1 CUDA GPU (for fast NN)
* 1 strong CPU with many threads (for simulating many environments at the
  same time).
* around 5 GB of hard drive space to download and extract the expert
      trajectories.

If you do not have a CPU with a lot of multi-threading, then you will probably
need to go down on number of environments (e.g.  `--num_env 16`), which
potentially could make gradients more noisy.

Please follow the below steps to run the experiments.

#### 0. Clone the project from github

```
git clone <project address from green button in the top right corner>
cd <newly created project root>
```

#### 1. Create a virtualenv for this project
Create a fresh virtualenv:
```
virtualenv -p python3 monte_env
```
Activate it:
```
source monte_env/bin/activate
```
#### 2. Install the required packages

First, install either CPU-based `tensorflow`:
```
pip install tensorflow==1.10.1
```
or if you have CUDA-enabled GPU and want to use it for this project (see
[TensorFlow documentation](https://www.tensorflow.org/install/) for more
details about TensorFlow GPU support):

```
pip install tensorflow-gpu==1.10.1
```

Then install the rest of the requirements:

```
pip install -r requirements.txt
```

Now you can:

#### a) Run training
*Note!* Here you will additionally need `ffmpeg` to write periodic evaluation videos.
Install it by executing:

- Mac              ```brew install ffmpeg```
- Ubuntu 14.04     ```sudo apt-get install libav-tools```
- for other newer versions of Ubuntu the following should work: ```sudo apt-get install ffmpeg```


To run the training, execute the following from within your virtualenv:

```
python -m baselines.acktr.run_atari_training
```

This will start the training process.  On a computer with i7 CPU and GTX 1080
GPU we see around 1500 fps.  To get results consistently beating the first
world, you will need to push around 200 M frames through the algorithm, so this
will take some time - around 36 hours @ 1500 fps.

You can watch the progress by watching the logs, which are written to
```
<project-root-dir>/openai-logs/<date-and-time-dependent-run-dir>
```
If you go to this directory, you can use the command:
```
watch -n1 'tail -n 3 *.monitor.csv'
```
to see various statistics of episodes in the sub-environments: episode lenghts,
final scores and so on:

```
==> 0.monitor.csv <==
8000.0,5626,6135.852456
8000.0,1723,6176.941117
5800.0,1796,6221.852446

==> 1.monitor.csv <==
8000.0,2819,5958.2911449
8000.0,3972,6055.5009089
8000.0,5346,6186.8718729

```

Above you see output from 2 environments (out of default 32) and in each row
the subsequent numbers represent episode reward, episode length in number steps
and time since training has begun.

Additionally, the system will periodically run 5 evaluation episodes, which will write:

- some performance stats: episode total rewards, lengths
- videos from evaluation episodes gameplay
- current policy network parameters.

You will find these in folders:

```
<project-root>/vid/...
```

This training code uses a pre-collected set of expert trajectories.  Currently,
for the convenience of first-time users of this repo, the default behavior is
to download our set of expert trajectories.  However, you could potentially use
your own trajectories, by changing constants in `run_atari_training.py`.
By default, these trajectories are stored in folder:

```
<project-root>/in_data/
```


#### b) Run example trained model to see some good evaluations
```
python -m baselines.acktr.run_eval --model models/cool_model.npy
```

This will load a pretrained model supplied with this repository.  You should
expect to see a screen pop up, where the neural net agent is going to play the
game. It should clear the first world (as taught by the expert) and pass some
part of the second world.

The pre-trained model is stored in `<project-root>/models/cool_model.npy`.
If you run training script, it will periodically write policy parameters.
You can use the current script, passing the writen policy as `--model`
parameter, to see how well they are playing.
