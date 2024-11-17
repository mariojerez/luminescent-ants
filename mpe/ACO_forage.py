import forage_v0 as forage_v0

env = forage_v0.parallel_env(render_mode="human")
observations, infos = env.reset()
env.render()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()