import os 

class BaseArgs:
    env_name: str = "h1hand-stand-v0"
    """the id of the environment"""
    agent: str = "fasttd3"
    """the agent to use: currently support [fasttd3, fasttd3_simbav2]"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_rank: int = 0
    """the rank of the device"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    project: str = "FastTD3"
    """the project name"""
    use_wandb: bool = True
    """whether to use wandb"""
    checkpoint_path: str = None
    """the path to the checkpoint file"""
    num_envs: int = 128
    """the number of environments to run in parallel"""
    num_eval_envs: int = 128
    """the number of evaluation environments to run in parallel (only valid for MuJoCo Playground)"""
    total_timesteps: int = 150000
    """total timesteps of the experiments"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""
    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""
    critic_learning_rate_end: float = 3e-4
    """the learning rate of the critic at the end of training"""
    actor_learning_rate_end: float = 3e-4
    """the learning rate for the actor at the end of training"""
    buffer_size: int = 1024 * 50
    """the replay memory buffer size"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32768
    """the batch size of sample from the replay memory"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    std_min: float = 0.001
    """the minimum scale of noise"""
    std_max: float = 0.4
    """the maximum scale of noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 2
    """the number of updates to perform per step"""
    init_scale: float = 0.01
    """the scale of the initial parameters"""
    num_atoms: int = 101
    """the number of atoms"""
    v_min: float = -250.0
    """the minimum value of the support"""
    v_max: float = 250.0
    """the maximum value of the support"""
    critic_hidden_dim: int = 1024
    """the hidden dimension of the critic network"""
    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    critic_num_blocks: int = 2
    """(SimbaV2 only) the number of blocks in the critic network"""
    actor_num_blocks: int = 1
    """(SimbaV2 only) the number of blocks in the actor network"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    eval_interval: int = 5000
    """the interval to evaluate the model"""
    render_interval: int = 5000
    """the interval to render the model"""
    compile: bool = True
    """whether to use torch.compile."""
    obs_normalization: bool = True
    """whether to enable observation normalization"""
    reward_normalization: bool = False
    """whether to enable reward normalization"""
    max_grad_norm: float = 0.0
    """the maximum gradient norm"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""
    disable_bootstrap: bool = False
    """Whether to disable bootstrap in the critic learning"""

    use_domain_randomization: bool = False
    """(Playground only) whether to use domain randomization"""
    use_push_randomization: bool = False
    """(Playground only) whether to use push randomization"""
    use_tuned_reward: bool = False
    """(Playground only) Use tuned reward for G1"""
    action_bounds: float = 1.0
    """(IsaacLab only) the bounds of the action space (-action_bounds, action_bounds)"""
    task_embedding_dim: int = 32
    """the dimension of the task embedding"""

    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    save_interval: int = 5000
    """the interval to save the model"""

