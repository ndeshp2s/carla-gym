class Config:
    env: str = None
    hyperparameters = None
    state_dim: int = None
    action_dim: int = None
    use_cuda = None
    number_of_episodes = 0
    steps_per_episode = 0
    total_steps = 0
    current_episode = 0
    log_dir = 'experiments/dqn_lstm/logs' 


