import forage_v0 as forage_v0
import numpy as np

export_data = False
data_file_name = "forage_0.XX_decay_XXX_range_trialX.csv"
masks = {}
buffer = 50 #cm
env = forage_v0.parallel_env(render_mode="human")
# deploy agents and resoures randomly, set initial luminescence and local-decision domain radius
observations, infos = env.reset()
env.render()
data = [] #["amount foraged", "random steps", "resources discovered", "resources foraged"]

while env.agents:
    # this is where you would insert your policy
    env.aec_env.update_luminescence()
    env.aec_env.update_neighbors() # neighbors = {agenti: [(neighborj, luminescencej), ...]}
    
    # Prevent agents from leaving environment boundaries or running into eachother
    for agent in observations:
        # Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`
        x_pos, y_pos = observations[agent][2], observations[agent][3]
        masks[agent] = np.array([1, # no action
                       1 if x_pos > buffer else 0, # move left
                       1 if x_pos < env.aec_env.length - buffer else 0, #move right
                       1 if y_pos > buffer else 0, # move down
                       1 if y_pos < env.aec_env.width - buffer else 0], dtype=np.int8) # move up
    
    actions = {agent: env.aec_env.choose_action(agent, mask=masks[agent]) for agent in env.agents}
    # obervation space of agents = Box(low=-inf, high=inf, shape=(20,), dtype=float32)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.aec_env.update_decision_range()
    if export_data:
        data.append([env.aec_env.world.nests[0].state.amount, env.aec_env.num_random_steps, len(env.aec_env.resources_discovered), len(env.aec_env.resources_foraged), len(env.aec_env.resources_depleted)])
    

print(f"Foraged: {env.aec_env.world.nests[0].state.amount}")
print(f"Number of random steps: {env.aec_env.num_random_steps}")
print(f"Resources discovered: {len(env.aec_env.resources_discovered)}")
print(f"Resources foraged: {len(env.aec_env.resources_foraged)}")
print(f"Resources depleted: {len(env.aec_env.resources_depleted)}")

if export_data:
    data = np.array(data)
    np.savetxt(data_file_name, data, delimiter=',', fmt='%d')
    env.close()