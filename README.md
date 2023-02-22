# Pheromones model
A model of 1d food search based on pheromones. The agent is walking in a circular channel. 
To be compared with 1d path integration fly models from the paper: [Drosophila re-zero their path integrator at the center of a fictive food patch](https://doi.org/10.1016/j.cub.2021.08.006)

### FR model (food to reward)
```fr_model.py```

FR model reproduced from Behbahani et al.
The fly resets the path integrator at food location and chooses a run length. At the reversal it chooses the new run length based on distribution of consequent run lengths differences. 

### Pheromone model
```pheromone_model.py```

The agent releases pheromone every time it encounters food. The pheromone has odor value which decreases over time. If the agent smells the pheromone, it selects the new run length based on the current odor value.
