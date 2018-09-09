# Expert-augmented actor-critic for Montezuma's Revenge

This repo contains code for Expert-augmented ACKTR algorithm that mixes
behavioral cloning and actor-critic to get good performance in Montezuma's
Revenge.

[Click here to see Arxiv paper.](https://arxiv.org/hehe)

Below you can see two excerpts from the gameplay.  On the right, agent exploits
an unreported bug in Montezuma's Revenge.  On the left, you can see part of
typical evaluation in the second world.

![No bug gameplay](https://ghostfacekillah.github.io/img/expert/life.gif) ![bug gameplay](https://ghostfacekillah.github.io/img/expert/bug.gif)

The algorithm is relatively easy to understand and implement.
The core of it is expressed in only one formula, see below:

![loss](https://ghostfacekillah.github.io/img/expert/loss-expression.png)

On the left, you can see the standard actor-critic A2C loss function. On the right,
you can see the new loss term added by the algorithm, which is similar in
spirit to the former one, but expectations are computed over batches of expert
transitions sampled from a fixed dataset.

Below you can see high-level graphical overview of the algorithm:

![stuff](https://ghostfacekillah.github.io/img/expert/model.png)

This code is based on [OpenAI's baselines](https://github.com/openai/baselines).
We are grateful to the authors of this repository.


### How to run the experiments?

The optimal hardware for running these experiments is 1 CUDA GPU (for fast NN) + 1
strong CPU with many threads (for simulating many environments at the same
time).

Please follow the below steps to run the experiments.

#### 1. Make a fresh virtualenv for this project
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
pip install tensorflow
```
or if you have CUDA-enabled GPU and want to use it for this project:
```
pip install tensorflow-gpu
```

Then install the rest of the requirements:

```
pip install -r requirements.txt
```

Now you can:

#### a) Run training
*Note!* You will need `ffmpeg` to write periodic evaluation videos.
Install it by executing:
  - Mac              `brew install ffmpeg`
  - Ubuntu 14.04     `sudo apt-get install libav-tools`
  - All other Ubuntus `sudo apt-get install ffmpeg`


```
python -m baselines.acktr.run_atari_training
```

This will start the training process.
You should expect to see around 1500 fps.
To get results consistently beating the first world, you will need to push around
200 M frames through the algorithm, so this will take some time - around 36 hours.

You can watch the progress by watching the logs, which are written to
```
<project-root-dir>/openai-logs/<date-and-time-dependent-run-dir>
```
If you go to this directory, you can use command
```
watch -n1 'tail -n 3 *.monitor.csv'
```
to see various statistics of episodes in the sub-environments: episode lenghts,
final scores, etc..


#### b) Run example trained model to see some good evaluations
```
python -m baselines.acktr.run_eval --model models/cool_model.npy
```

This will load a pretrained model supplied with this repository.
You should expect to see a screen pop up, where the neural net agent is
going to play the game. It should clear the first world (as taught by the expert)
and pass some part of the second world.

