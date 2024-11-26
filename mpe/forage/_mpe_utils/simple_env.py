import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from forage._mpe_utils.core import Agent, Resource
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        dynamic_rescaling=False,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.length = 1246 # cm
        self.width = 639 #cm
        self.screen = pygame.Surface([self.length, self.width])
        self.max_size = 1 #TODO: See what this does
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio
        self.dynamic_rescaling = dynamic_rescaling

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self.observed_resources = dict()
        self.neighbors = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1 # if dim_p is 2, space_dim = 5. Why? TODO: Understand what space_dim is for
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions: #TODO: Decide if we want to have continuous actions
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim) #n = space_dim, start = 0 (num elements in space) #for some reason action spaces is dim_0 * 2 + 1 + obs_dim
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Get the original cam_range
        # This will be used to scale the rendering
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        self.original_cam_range = np.max(np.abs(np.array(all_poses))) #TODO: Get rid of this


        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def update_luminescence(self, decay=0.2, growth=0.5):
        """
        Update luminescence value for each agent

        Parameters:
            decay (float): luminescence decay constant (0,1)
            growth (float): proportionality constant for enhancing the luminescence as a function of the objective/fitness function
        """
        self.update_objective_value()
        # TODO: Can do the following more efficiently using numpy array
        for agent in self.world.agents:
            agent.state.lum = np.max([0, (1-decay)*agent.state.lum + growth*agent.state.fit])

    def update_objective_value(self):
        """
        Update objective function values for all agent, which is defined as the amount of food within an agent's decision-domain.
        Choosing decision domain instead of radial sensor range because agents surrounded by fewer agents should signal more loudly
        than agents surrounded by many agents.
        """
        for agent in self.world.agents:
            resources_in_domain = []
            fit = 0
            for resource in self.world.resources:
                agent_to_resource = resource.state.p_pos - agent.state.p_pos
                dist = np.sqrt(np.sum(np.square(agent_to_resource)))
                if dist <= agent.state.decision_domain:
                    resources_in_domain.append(resource)
                    fit += resource.state.amount
            self.observed_resources[agent] = resources_in_domain
            agent.state.fit = fit

    def update_neighbors(self):
        neighbors = {agent: [] for agent in self.world.agents}
        # TODO: Do more efficiently using numpy array
        for agent in self.world.agents:
            for neighbor in self.world.agents:
                if neighbor == agent:
                    continue
                agent_to_neighbor = neighbor.state.p_pos - agent.state.p_pos
                dist = np.sqrt(np.sum(np.square(agent_to_neighbor)))
                if dist <= agent.state.decision_domain:
                    neighbors[agent].append((neighbor))
        self.neighbors = neighbors
            
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action ## TODO: Handle step size, handle masking so they can't go off grid
                #TODO: Check that allows themto move diagonally.
                if action[0] == 1:
                    agent.action.u[0] = -100.0
                if action[0] == 2:
                    agent.action.u[0] = +100.0
                if action[0] == 3:
                    agent.action.u[1] = -100.0
                if action[0] == 4:
                    agent.action.u[1] = +100.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2)) #
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        #all_poses = [entity.state.p_pos for entity in self.world.entities]
        #cam_range = np.max(np.abs(np.array(all_poses)))

        #TODO: Get rid of scaling
        #TODO: Adjust environment so that it is bounded between -1 and 1

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds
        #scaling_factor = 0.9 * self.original_cam_range / cam_range

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            font = pygame.font.Font('mpe/forage/_mpe_utils/secrcode.ttf', 20)

            if isinstance(entity, Agent):
                pygame.draw.circle(self.screen, entity.color * 200, (x, y), entity.size)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.size, 1) # border
                # show decision domain
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.state.decision_domain, 1)

                # show state information
                text = font.render(f"τ: {round(entity.state.lum, 3)}", False, (0,0,0))
                self.screen.blit(text, (x,y+20))


            elif isinstance(entity, Resource):
                pygame.draw.rect(self.screen, entity.color * 200, (x, y, entity.size*2, entity.size*2))
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, entity.size*2, entity.size*2), 1)
                text = font.render(f"amount: {entity.state.amount}", False, (0,0,0))
                self.screen.blit(text, (x,y+20))

            #assert (
            #    0 < x < self.width and 0 < y < self.height
            #), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                #TODO: Maybe use message to broadcast luminescence
                message = entity.name + " sends " + word + "   "
                message_x_pos = self.length * 0.05
                message_y_pos = self.width * 0.95 - (self.width * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None