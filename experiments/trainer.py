from experiments.config import Config

class Trainer:
	def __init__(self, env, agent, config: Config):
		self.env = env
		self.agent = agent
		self.config = config

	def train(self, previous_episode = 0):
		losses = []
		rewards = []
		
		epsilon = self.config.hyperparamaters["epsilon_start"]

		for e in range(previous_episode + 1, self.congif.number_of_episodes + 1):
			episode_reward = 0
			state = env.reset()

			for s in range(self.config.steps_per_episode):
				action = None
				next_state, reward, done, _ = None

				loss = self.agent.step(state, action, reward, next_state, done)

				if loss is not None:
					losses.append(loss)
					self.tensorboard_logger.scalar_summary('Loss per step', s, loss)

				if done:
					break

				state = next_state

				episode_reward += reward


