# behbahani_model
Reproduce 1d path integration fly models from the paper: [Drosophila re-zero their path integrator at the center of a fictive food patch](https://doi.org/10.1016/j.cub.2021.08.006)

### FR model (food to reward)
```fr_model.py```

The fly resets the integrator at food location and chooses a run length. At the reversal it chooses the new run length based on distribution of consequent run lengths differences. 
