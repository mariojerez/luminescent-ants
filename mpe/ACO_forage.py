import forage_v0 as forage_v0

env = forage_v0.parallel_env(render_mode="human")
observations, infos = env.reset()
env.render()

while env.agents:
    # this is where you would insert your policy
    # actions can be 0, 1, 2, 3, or 4
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} #randomly chooses from action space. Can include mask to prevent movement in certain direction
    # obervation space of agents = Box(low=-inf, high=inf, shape=(20,), dtype=float32)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # resource1 = env.aec_env.world.resources[0]
    # resource2 = env.aec_env.world.resources[1]
    # print(f"{resource1.name} position: {resource1.state.p_pos}, velocity: {resource1.state.p_vel}; {resource2.name} position: {resource2.state.p_pos}, velocity: {resource2.state.p_vel}")
env.close()