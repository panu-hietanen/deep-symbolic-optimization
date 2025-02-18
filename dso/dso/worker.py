import tensorflow as tf
import numpy as np
import multiprocessing as mp

from dso.program import Program, from_tokens
from dso.tf_state_manager import HierarchicalStateManager, HierarchicalStateManager


class Worker(mp.Process):
    def __init__(
            self,
            worker_id,
            policy_class,  # e.g. RNNPolicy
            prior,
            policy_kwargs,  # dict of arguments for your policy constructor
            state_manager_kwargs,
            task_queue,  # receives commands like {"type": "sample", "batch_size": ...}
            result_queue,  # used to send sampled batches (or other data) back
            param_queue,  # used to receive updated parameters from the main process
            batch_size,
    ):
        super().__init__()
        # tf.reset_default_graph()
        self.sess = None
        self.policy = None
        self.state_manager = None

        self.worker_id = worker_id
        self.policy_class = policy_class
        self.prior = prior
        self.policy_kwargs = {key: value for key,value in policy_kwargs.items() if key != 'policy_type'}
        self.state_manager_kwargs = {key: value for key,value in state_manager_kwargs.items() if key != 'type'}
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.param_queue = param_queue
        self.batch_size = batch_size

    def run(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        if self.sess is None:
            self.sess = tf.Session(config=config)
        if self.state_manager is None:
            self.state_manager = HierarchicalStateManager(**self.state_manager_kwargs)

        if self.policy is None:
            self.policy = self.policy_class(
                self.sess,
                self.prior,
                self.state_manager,
                self.worker_id,
                **self.policy_kwargs
            )

        self.sess.run(tf.global_variables_initializer())

        while True:
            task = self.task_queue.get()
            if task is None:
                print(f"Worker {self.worker_id} stopping...")
                break

            elif task["type"] == "update_params":
                new_params = task["params"]
                self.set_params(new_params)

            elif task["type"] == "sample":
                override = task["override"]
                actions, obs, priors, programs, n_extra = self.sample_batch(override)
                data = {
                    "worker_id": self.worker_id,
                    "actions": actions,
                    "obs": obs,
                    "priors": priors,
                    "programs": programs,
                    "n_extra": n_extra,
                }
                self.result_queue.put(data)

            else:
                raise NotImplementedError(f"Worker {self.worker_id} received unknown task type: {task['type']}")

        self.sess.close()

    def sample_batch(self, override):
        """
        Sample a batch of expressions from the policy, or if override is provided,
        use the override data.
        Returns (actions, obs, priors, programs) for the next training iteration.
        """
        if override is None:
            # Sample batch of Programs from the Controller
            actions, obs, priors = self.policy.sample(self.batch_size)
            programs = [from_tokens(a) for a in actions]
        else:
            # Train on the given batch of Programs
            actions, obs, priors, programs = override

        n_extra = 0

        # Possibly add extended batch
        if self.policy.valid_extended_batch:
            self.policy.valid_extended_batch = False
            n_extra = self.policy.extended_batch[0]
            if n_extra > 0:
                extra_programs = [from_tokens(a) for a in self.policy.extended_batch[1]]
                actions = np.concatenate([actions, self.policy.extended_batch[1]])
                obs = np.concatenate([obs, self.policy.extended_batch[2]])
                priors = np.concatenate([priors, self.policy.extended_batch[3]])
                programs += extra_programs
        return actions, obs, priors, programs, n_extra

    def set_params(self, params):
        """
        Set the parameters of the policy according to broadcast
        from main process.
        """
        self.policy.set_params(params, self.worker_id)