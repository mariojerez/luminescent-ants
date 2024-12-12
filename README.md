# Foraging with Glowworm Swarm Based Algorithm in pettingzoo
An Ant Colony Optimization model of muliple agents simulated in PettingZoo

This was built from of the Simple Spread environment in the pettingzoo library.
https://pettingzoo.farama.org/environments/mpe/simple_spread/

To see the algorithm in action, run mpe/ACO_forage.py or

watch a video: https://youtu.be/-sATLyRs4Sg

## High Level Description
Agents and resources get randomly distributed round the map. The objective is for agents to find resources and bring them to the nest (in the center of the map). They can communicate through luminescence values, similarly to how ants communicate through pheromones. When an agent is surrounded by lots of resources, they have a higher luminescence value. If there are agents within agent i's decision domain, agent i will follow one of them if their luminescence values are greater than that of agent i; otherwise, agent i will go in a random direction. The probability of agent i following another agent is proportional to the strength of that agent's luminescence value.

Agents can exhibit two behaviors: explore and forage. All agents start off as explorers, where they follow K. N. Krishnanand and D. Ghose' glowworm swarm based algorithm [1]. Once they reach a resource, they stop and keep emitting their luminescence values (which keep getting updated). If an agent is within reach of a resource, and there is already another agent that is also within reach of that resource exhibiting explore behavior, than the agent switches to foraging behavior where it "picks up" a unit of that resource and takes it to the nest. After leaving it at the nest, it exhibits explore behavior once again.


[1] K. N. Krishnanand and D. Ghose, “Detection of multiple source locations using a glowworm metaphor with applications to collective
robotics,” in Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005., Conference Proceedings, pp. 84–91. [Online].
Available: https://ieeexplore.ieee.org/document/1501606/

## Integration with Turtlebots in Gazebo
See the ROS_integration branch

The ROS code can be found here: https://github.com/jackswanberg/ROS_Ant_Colony
