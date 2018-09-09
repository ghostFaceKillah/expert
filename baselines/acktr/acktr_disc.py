import joblib
import numpy as np
import tensorflow as tf
import time

from baselines import logger
from baselines.acktr import kfac
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.acktr.utils import cat_entropy, mse
from baselines.acktr.utils import discount_with_dones
from baselines.common import set_global_seeds, explained_variance
from baselines.common.expert import ExpertRunner
from baselines.video.video_runners import easy_video


class Runner(object):
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)

        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0

            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs,
                 expert_nbatch,
                 total_timesteps,
                 nprocs=32, nsteps=20,
                 ent_coef=0.01,
                 vf_coef=0.5, vf_fisher_coef=1.0, vf_expert_coef=0.5 * 0.0,
                 expert_coeff=1.0,
                 exp_adv_est='reward',
                 lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear'):

        # create tf stuff
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)

        # the actual model
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        A_EXP = tf.placeholder(tf.int32, [expert_nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        ADV_EXP = tf.placeholder(tf.float32, [expert_nbatch])

        R = tf.placeholder(tf.float32, [nbatch])
        R_EXP = tf.placeholder(tf.float32, [expert_nbatch])

        PG_LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        eval_step_model = policy(sess, ob_space, ac_space, 1, 1, reuse=True)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)
        expert_train_model = policy(sess, ob_space, ac_space, expert_nbatch, 1, reuse=True)
        logpac_expert = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=expert_train_model.pi, labels=A_EXP)
        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)

        _, acc = tf.metrics.accuracy(labels=A,
                                     predictions=tf.argmax(train_model.pi, 1))

        ## training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        pg_expert_loss = tf.reduce_mean(ADV_EXP * logpac_expert)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        vf_expert_loss = tf.reduce_mean(mse(tf.squeeze(expert_train_model.vf), R_EXP))
        train_loss = pg_loss + vf_coef * vf_loss + expert_coeff * pg_expert_loss + vf_expert_coef * vf_expert_loss

        self.check = check = tf.add_check_numerics_ops()

        ## Fisher loss construction
        pg_fisher_loss = -tf.reduce_mean(logpac)  # + logpac_expert)
        # pg_expert_fisher_loss = -tf.reduce_mean(logpac_expert)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        vf_fisher_loss = - vf_fisher_coef * tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(
                learning_rate=PG_LR, clip_kl=kfac_clip,
                momentum=0.9, kfac_update=1, epsilon=0.01,
                stats_decay=0.99, async=1, cold_iter=20, max_grad_norm=max_grad_norm
            )

            # why is this unused?
            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values,
                  expert_obs, expert_rewards, expert_actions, expert_values):
            if exp_adv_est == 'critic':
                expert_advs = np.clip(expert_rewards - expert_values, a_min=0, a_max=None)
            elif exp_adv_est == 'reward':
                expert_advs = expert_rewards
            elif exp_adv_est == 'simple':
                expert_advs = np.ones_like(expert_rewards)
            else:
                raise ValueError("Unknown expert advantage estimator {}".format(exp_adv_est))

            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {
                train_model.X:obs,
                expert_train_model.X: expert_obs,
                A_EXP: expert_actions,
                A:actions,
                ADV:advs,
                ADV_EXP: expert_advs,
                R:rewards,
                PG_LR:cur_lr,
                R_EXP: expert_rewards
            }

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, policy_expert_loss, value_loss, policy_entropy, train_accuracy, _, grads_to_check = sess.run(
                [pg_loss, pg_expert_loss, vf_loss, entropy, acc, train_op, grads],
                td_map
            )

            for grad in grads_to_check:
                if np.isnan(grad).any():
                    print("ojojoj grad is nan")

            return policy_loss, policy_expert_loss, value_loss, policy_entropy, train_accuracy

        def save(save_path):
            print("Writing model to {}".format(save_path))
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        def eval_step(obs, eval_type):
            td_map = {eval_step_model.X: [obs]}
            logits = sess.run(eval_step_model.pi, td_map)[0]
            if eval_type == 'argmax':
                act = logits.argmax()
                if np.random.rand() < 0.01:
                    act = ac_space.sample()
                return act
            elif eval_type == 'prob':
                # probs = func(s[None, :, :, :])[0][0]
                x = logits
                e_x = np.exp(x - np.max(x))
                probs = e_x / e_x.sum(axis=0)
                act = np.random.choice(range(probs.shape[-1]), 1, p=probs)[0]
                return act
            else:
                raise ValueError("Unknown eval type {}".format(eval_type))

        self.model = step_model
        self.model2 = train_model
        self.expert_train_model = expert_train_model
        self.vf_fisher = vf_fisher_loss
        self.pg_fisher = pg_fisher_loss
        self.joint_fisher = joint_fisher_loss
        self.params = params
        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.eval_step = eval_step
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)


def learn(policy, env, seed, params,
          dataflow_config,
          expert_nbatch,
          exp_adv_est='reward',
          load_model=None,
          total_timesteps=int(40e6),
          gamma=0.99,
          nprocs=32, nsteps=20,
          ent_coef=0.01,
          vf_coef=0.5, vf_fisher_coef=1.0, vf_expert_coef=0.5 * 0.0,
          expert_coeff=1.0,
          lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, lrschedule='linear', video_interval=3600):

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(
        policy, ob_space, ac_space, nenvs,
        expert_nbatch,
        total_timesteps,
        nprocs=nprocs, nsteps=nsteps, ent_coef=ent_coef,
        vf_coef=vf_coef, vf_expert_coef=vf_expert_coef, vf_fisher_coef=vf_fisher_coef,
        lr=lr, max_grad_norm=max_grad_norm,
        kfac_clip=kfac_clip, lrschedule=lrschedule,
        expert_coeff=expert_coeff,
        exp_adv_est=exp_adv_est,
    )

    if load_model is not None and load_model != 'None':
        print("Loading model {}".format(load_model))
        model.load(load_model)

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    expert_runner = ExpertRunner(env, model, dataflow_config)

    nbatch = nenvs*nsteps
    tstart = time.time()

    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)

    t_last_update = tstart
    t_last_vid = 0

    update = 0


    while True:
        update += 1
        obs, states, rewards, masks, actions, values = runner.run()
        exp_obs, exp_actions, exp_rewards, exp_values = expert_runner.run()

        policy_loss, policy_expert_loss, value_loss, policy_entropy, train_accuracy = model.train(
            obs, states, rewards, masks, actions, values,
            exp_obs, exp_rewards, exp_actions, exp_values
        )

        now = time.time()

        if now - t_last_update > 10:
            nseconds = now - tstart

            if np.abs(expert_coeff) < 1e-9:
                # if we don't use expert data, we don't count it
                nframes = update * nbatch
            else:
                nframes = update * (nbatch + expert_nbatch)

            fps = int(float(nframes)/nseconds)
            mframes = nframes / 1e6

            t_last_update = now
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("mframes", mframes)
            logger.record_tabular("fps", fps)
            logger.record_tabular("expert_train_accuracy", float(train_accuracy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("policy_expert_loss", float(policy_expert_loss))
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))

            logger.dump_tabular()

        if now - t_last_vid > video_interval:
            easy_video(model, params, 'prob')
            t_last_vid = time.time()

    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
