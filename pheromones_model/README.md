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

## 3 rewards experiment

``` 
python generate_3foods_trajectories.py three_rewards_5.csv 
python plot_distributions_rew3.py data/rewards3/three_rewards_5.csv
```

Config file for generation: `config_rew3.yaml`

The csv filename (`three_rewards_5.csv`) has to be in the model_settings section of the config.
The data will be saved to a folder specified under data_folder entry of the config (in this case data/rewards3). This directory has to exist.
`plot_distributions_rew3.py` creates the plots and processed csv files.

## Circling experiment

``` 
python generate_circling_trajs.py circling_simulations5.csv
python circling_analyze.py data/circling/circling_simulations5.csv
python circling_nice_plot.py data/circling/circling_simulations5.csv
```

Config file for generation: `config_circling.yaml`

The csv filename (`circling_simulations5.csv`) has to be in the model_settings section of the config.

The data will be saved to a folder specified under data_folder entry of the config (in this case data/circling). 
This directory has to exist.

`circling_analyze.py` creates processed csv files and a pdf with plots next to the generated data file (it is necessary to specify relative path as a command line argument).

`circling_nice_plot.py` creates a combined plot in a svg file. Needs to be run after `circling_analyze.py`.
