# noqa: D212, D415
"""
# Switch Sides
TODO: update comments below to be accurate to switch_sides rather than simple_spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from forage._mpe_utils.core import Agent, Resource, World
from forage._mpe_utils.scenario import BaseScenario
from forage._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=4,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio
        )
        self.metadata["name"] = "switch_sides_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents=4, num_resources=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_resources = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True # TODO: Decide if will set to False
            agent.silent = True
            agent.size = 0.05
        # add resources
        world.resources = [Resource() for i in range(num_resources)] #TODO: Adjust Landmark code to Food
        for i, resource in enumerate(world.resources):
            resource.name = "resource %d" % i
            resource.collide = True
            resource.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for resources
        for i, resource in enumerate(world.resources):
            resource.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p) # positional dimensions
            agent.state.c = np.zeros(world.dim_c) # color dimensions
        for i, resource in enumerate(world.resources): 
            resource.state.p_pos = np_random.uniform(-1, +1, world.dim_p) # TODO: Check if something other than uniform can be used to clump food together
            resource.state.p_vel = np.zeros(world.dim_p) #TODO: Should not have option for velocity

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_resources = 0
        min_dists = 0
        for r in world.resources:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - r.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_resources += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_resources)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world): #TODO: Adapt for objective function, which counts how much food is within decision domain
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
        return rew

    def global_reward(self, world):
        rew = 0
        for r in world.resources:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - r.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    #TODO: Add local-decison domain (and radial sensor range) instance variable to agent and depict it with an outer outline in pygame.
    def observation(self, agent, world): #TODO: Edit so that can only observe agents within local-decision domain
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for resource in world.resources:  # world.entities:
            entity_pos.append(resource.state.p_pos - agent.state.p_pos) # I think returns vector of length distance between entity and landmark and direction of landmark. TODO: ADAPT FOR vector to other ants and food (within local-decision domain)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos) #gets vector from agent to something else that isn't an agent?
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )
        # returns agent's velocity and position vectors, the relative positions of landmarks, and relative pos of other agents, and comm is all 0