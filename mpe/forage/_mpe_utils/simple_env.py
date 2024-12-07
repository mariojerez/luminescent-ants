import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from forage._mpe_utils.core import Agent, Resource, Nest
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
        continuous_actions=True,
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
        self.max_size = 1
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
        self.detected_food = dict()
        self.food_within_reach = dict()
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
                    low=0, high=101, shape=(space_dim,)
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
        return self.action_spaces[agent] # warning happens here

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
        """Update neighboring agents and resources"""
        self.neighbors = {agent: [] for agent in self.world.agents}
        self.detected_food = {agent: [] for agent in self.world.agents}
        self.food_within_reach = {agent: [] for agent in self.world.agents}
        for agent in self.world.agents:
            # Update agent neighbors
            for neighbor in self.world.agents:
                if neighbor == agent or neighbor.state.lum <= agent.state.lum or neighbor.state.behavior == "forage":
                    continue
                agent_to_neighbor = neighbor.state.p_pos - agent.state.p_pos
                dist = np.sqrt(np.sum(np.square(agent_to_neighbor)))
                if dist <= agent.state.decision_domain:
                    self.neighbors[agent].append(neighbor)

            # Update resource neighbors
            for resource in self.world.resources:
                if resource.state.amount <= 0:
                    continue
                dist = self.dist(agent.state.p_pos, resource.state.p_pos)
                if dist <= agent.food_detection_range:
                    self.detected_food[agent].append(resource)
                if dist <= agent.reach:
                    self.food_within_reach[agent].append(resource)

    def dist(self, location_1, location_2):
        return np.sqrt(np.sum(np.square(location_2 - location_1)))

    def choose_action(self, agent):
        self.set_behavior_type(agent)
        if agent.state.behavior == "explore":
            neighbor = self.choose_neighbor(agent) #TODO: Have choose_neighbor update an instance variable rather than returning a neighbor
            if len(self.food_within_reach[agent]) > 0:
                # don't move
                return np.array([1.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
            elif len(self.detected_food[agent]) > 0:
                resource_amount = np.array([resource.state.amount for resource in self.detected_food[agent]])
                resource = self.detected_food[agent][np.argmax(resource_amount)]
                x, y = resource.state.p_pos - agent.state.p_pos
            elif neighbor is None:
                if agent.state.heading is None:
                    return self.action_spaces[agent.name].sample()
                else:
                    # adjust agent heading by up to 30 degrees
                    angle = np.radians(np.random.uniform(-30, 30))
                    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                                [np.sin(angle), np.cos(angle)]])
                    x, y = np.dot(rotation_matrix, agent.state.heading)
            else:
                x, y = neighbor.state.p_pos - agent.state.p_pos
        elif agent.state.behavior == "forage":
            if agent.state.carrying == "":
                assert len(self.food_within_reach[agent]) > 0, f"expected for agent {agent.name} to have food within reach, but it did not."
                resource = self.food_within_reach[agent][0]
                assert resource.state.amount > 0, f"expected resource amount > 0, but it is {resource.state.amount}."
                agent.state.carrying = resource.name
                resource.state.amount -= 1
                # if resource depleted, remove from self.food_within_reach and self.detected_food
                if resource.state.amount <= 0:
                    for agent in self.world.agents:
                        if resource in self.food_within_reach[agent]:
                            self.food_within_reach[agent].remove(resource)
                        if resource in self.detected_food[agent]:
                            self.detected_food[agent].remove(resource)

            nest = self.world.nests[0]
            if self.dist(agent.state.p_pos, nest.state.p_pos) < nest.size:
                # leave resoruce at nest
                nest.state.amount += 1
                agent.state.carrying = ""
                agent.state.behavior = "explore"
            # go towards nest
            x, y = nest.state.p_pos - agent.state.p_pos
        else:
            raise Exception("expected behavior type explore or forage, but received " + agent.state.behavior)
        x, y = np.array([x,y]) * (1 / np.sqrt(np.sum(np.square(np.array([x,y]))))) # turn to unit vector. Magnitude gets adjusted later.
        return np.array([0, -1 * x if x < 0 else 0, x if x > 0 else 0, -1 * y if y < 0 else 0, y if y > 0 else 0]).astype(np.float32) # [no_action, move_left, move_right, move_down, move_up]
            
    def set_behavior_type(self, agent):
        if agent.state.carrying != "":
            agent.state.behavior = "forage"
            return
        elif len(self.food_within_reach[agent]) > 0:
            for resource in self.food_within_reach[agent]:
                if not self.resource_has_signaller(resource, agent):
                    agent.state.behavior = "explore"
                    return
            agent.state.behavior = "forage"
            return
        agent.state.behavior = "explore"

    def resource_has_signaller(self, resource, other_than_agent):
        for agent in self.food_within_reach:
            if agent == other_than_agent:
                continue
            if resource in self.food_within_reach[agent] and agent.state.behavior == "explore":
                return True
        return False

    def choose_neighbor(self, agent):
        if len(self.neighbors[agent]) == 0:
            return None
        neighbor_lum = np.zeros(len(self.neighbors[agent]))

        for i in range(len(self.neighbors[agent])):
            neighbor_lum[i] = self.neighbors[agent][i].state.lum

        # set to probabilities
        neighbor_lum = neighbor_lum / np.sum(neighbor_lum)
        rand = np.random.rand()
        prob_sum = 0
        for i in range(len(neighbor_lum)):
            prob_sum += neighbor_lum[i]
            if rand < prob_sum:
                return self.neighbors[agent][i]
            
    def update_decision_range(self):
        for agent in self.world.agents:
            neighbor_density = (len(self.neighbors[agent]) / (np.pi * agent.sensor_range**2)) * 100 # * 100 to give density in meters
            agent.state.decision_domain = agent.sensor_range / (1 + agent.beta * neighbor_density)

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
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                # [no_action, move_left, move_right, move_down, move_up]
                agent.action.u[0] += action[0][2] - action[0][1] # x = move_right - move_left
                agent.action.u[1] += action[0][4] - action[0][3] # y = move_up - move_down
            else:
                # process discrete action ## TODO: Handle step size, handle masking so they can't go off grid
                #TODO: Check that allows themto move diagonally.
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            # sensitivity = 50.0
            # if agent.accel is not None:
            #     sensitivity = agent.accel
            # agent.action.u *= sensitivity
            if np.sqrt(np.sum(np.square(agent.action.u))) > 0:
                # Set magnitude of force to 100
                agent.action.u *= 100 / np.sqrt(np.sum(np.square(agent.action.u)))
                agent.state.heading = agent.action.u / np.sqrt(np.sum(np.square(agent.action.u)))
                agent.ros_controller.update_velocity_petting_zoo(agent.action.u,agent.state.heading)
                # agent.state.heading = agent.ros_controller.get_orientation()
            else:
                #If not action stop robot motion
                agent.ros_controller.halt()
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
            font = pygame.font.Font(
                os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 20
            )
            # font = pygame.font.Font('mpe/forage/_mpe_utils/secrcode.ttf', 20)

            if isinstance(entity, Agent):
                pygame.draw.circle(self.screen, entity.color * 200, (x, y), entity.size)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.size, 1) # border
                # show decision domain
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.state.decision_domain, 1)

                # show state information
                text = font.render(f"{entity.name} behavior: {entity.state.behavior}\ncarrying: {entity.state.carrying}", False, (0,0,0))
                self.screen.blit(text, (x,y+20))

                # draw heading
                if entity.state.heading is not None:
                    end_heading_x, end_heading_y = entity.state.p_pos + entity.state.heading * 1.5 * entity.size
                    pygame.draw.line(self.screen, (255, 0, 0), (x, y), (end_heading_x, end_heading_y), 2)


            elif isinstance(entity, Resource):
                pygame.draw.rect(self.screen, entity.color * 200, (x, y, entity.size*2, entity.size*2))
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, entity.size*2, entity.size*2), 1)
                text = font.render(f"{entity.name} amount: {entity.state.amount}", False, (0,0,0))
                self.screen.blit(text, (x,y+20))

            elif isinstance(entity, Nest):
                pygame.draw.circle(self.screen, entity.color*200, (x, y), entity.size)
                text = font.render(f"amount: {entity.state.amount}", False, (0,0,0))
                self.screen.blit(text, (x-60,y-60))

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
        for agent in self.world.agents:
            agent.ros_controller.halt()
        if self.screen is not None:
            pygame.quit()
            self.screen = None