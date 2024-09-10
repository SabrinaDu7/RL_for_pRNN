# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RecACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cell, memory_size=300, with_obs=False, rgb=True, with_HD=False):
        super().__init__()

        # Decide which components are enabled
        self.memorysize = memory_size
        self.with_obs = with_obs
        self.rgb = rgb
        self.with_HD = with_HD

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        self.memory_rnn = cell(self.image_embedding_size, self.memorysize)

        # Define embedding size
        if self.with_obs:
            self.embedding_size = self.image_embedding_size + self.memorysize
        else:
            self.embedding_size = self.memorysize
        if self.with_HD:
            self.embedding_size += 1

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, memory, noise, **kwargs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        if self.rgb:
            x /= 255
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        memory, _ = self.memory_rnn(x, noise, memory)

        if self.with_obs:
            embedding = torch.cat((x, memory), dim=1)
        else:
            embedding = memory

        if self.with_HD:
            embedding = torch.cat((embedding, obs.direction.unsqueeze(dim=1)), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
    

class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space, with_HD=True, rgb=True):
        super().__init__()
        self.with_HD = with_HD
        self.rgb = rgb

        # Define image embedding
        self.CV(obs_space)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def embedding_size(self):
        if self.with_HD:
            return self.image_embedding_size + 1
        else:
            return self.image_embedding_size
    
    def CV(self, obs_space):
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

    def forward(self, obs, **kwargs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        if self.rgb:
            x /= 255
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.with_HD:
            embedding = torch.cat((x, obs.direction.unsqueeze(dim=1)), dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class ACModelSR(ACModel):
    def __init__(self, obs_space, action_space, SR_size=-1, with_CV=True, rgb=True):
        self.with_CV = with_CV
        self.SR_size = SR_size + 1 # if SRs are not used, the arg should be -1
        self.rgb = rgb
        super(ACModelSR, self).__init__(obs_space, action_space)

    @property
    def embedding_size(self):
        return self.image_embedding_size + self.SR_size
    
    def CV(self, obs_space):
        if self.with_CV:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            n = obs_space["image"][0]
            m = obs_space["image"][1]
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        else:
            self.image_embedding_size = 0

    def forward(self, obs, SR, **kwargs):
        if self.with_CV:
            x = obs.image.transpose(1, 3).transpose(2, 3)
            if self.rgb:
                x /= 255
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)

        if self.with_CV and self.SR_size:
            embedding = torch.cat((x, SR, obs.direction.unsqueeze(dim=1)), dim=1)
        elif self.SR_size:
            embedding = torch.cat((SR, obs.direction.unsqueeze(dim=1)), dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

