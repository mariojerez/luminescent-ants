#!/usr/bin/env python3
import forage_v0 as forage_v0
import rospy
import time

time.sleep(15) #Let Gazebo boot up before starting sim

env = forage_v0.parallel_env(render_mode="human")
# deploy agents and resoures randomly, set initial luminescence and local-decision domain radius
observations, infos = env.reset()
env.render()

while env.agents:
    # this is where you would insert your policy
    env.aec_env.update_luminescence()
    env.aec_env.update_neighbors() # neighbors = {agenti: [(neighborj, luminescencej), ...]}

    actions = {agent.name: env.aec_env.choose_action(agent) for agent in env.aec_env.world.agents} #randomly chooses from action space. Can include mask to prevent movement in certain direction
    # obervation space of agents = Box(low=-inf, high=inf, shape=(20,), dtype=float32)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.aec_env.update_decision_range()
env.close()