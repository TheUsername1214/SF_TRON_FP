import torch


class Agent_State_Buffer:
    def __init__(self, state_dim, actuator_num, agent_num, max_step, device):
        self.state_dim = state_dim
        self.actuator_num = actuator_num
        self.agent_num = agent_num
        self.max_step = max_step
        self.device = device
        self.state_buffer = torch.zeros((max_step, agent_num, state_dim), device=self.device)
        self.action_buffer = torch.zeros((max_step, agent_num, actuator_num), device=self.device)
        self.next_state_buffer = torch.zeros((max_step, agent_num, state_dim), device=self.device)
        self.reward_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)
        self.over_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)
        self.GAE_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)

    def compute_GAE(self, critic_net, gamma, lam):
        with torch.no_grad():
            target_value = self.reward_buffer + \
                           (1 - self.over_buffer) * gamma * critic_net(self.next_state_buffer)

            current_value = critic_net(self.state_buffer)

        GAE = (target_value - current_value)
        advantage = 0
        index = -1
        for delta in GAE.flip(0):
            advantage = gamma * lam * advantage * (1 - self.over_buffer[index]) + delta
            self.GAE_buffer[index] = advantage
            index += -1

        self.GAE_buffer = (self.GAE_buffer - self.GAE_buffer.mean()) / self.GAE_buffer.std()

    def store_state(self, current_state, current_step):
        self.state_buffer[current_step] = current_state

    def store_action(self, current_action, current_step):
        self.action_buffer[current_step] = current_action

    def store_next_state(self, next_state, current_step):
        self.next_state_buffer[current_step] = next_state

    def store_reward(self, current_reward, current_step):
        self.reward_buffer[current_step] = current_reward

    def store_over(self, current_over, current_step):
        self.over_buffer[current_step] = current_over
