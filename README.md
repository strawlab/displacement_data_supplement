Dataset of displacement experiments providing evidence for path integration in Drosophila
---

Data pertaining to and described in the publication "Displacement experiments provide evidence for path integration in Drosophila" https://doi.org/10.1242/jeb.245289

### Publication abstract
Like many other animals, insects are capable of returning to previously visited locations using path integration,
which is a memory of travelled direction and distance. Recent studies suggest that Drosophila can also use path integration
to return to a food reward.
However, the existing experimental evidence for path integration in Drosophila has a potential confound:
pheromones deposited at the site of reward might enable flies to find previously rewarding locations even without
memory. Here, we show that pheromones can indeed cause naïve flies to accumulate where previous flies had been rewarded
in a navigation task. Therefore, we designed an experiment to determine if flies can use path integration memory
despite potential pheromonal cues by displacing the flies shortly after an optogenetic reward.
We found that rewarded flies returned to the location predicted by a memory-based model.
Several analyses are consistent with path integration as the mechanism by which flies returned to the reward.
We conclude that although pheromones are often important in fly navigation and must be carefully controlled for in future
experiments, Drosophila may indeed be capable of performing path integration.

### Experiments procedure overview
There were two independent experiments: the pheromone experiment and the displacement experiment.

In the pheromone experiment the trajectories of groups of flies walking in the circular 20 cm diameter arena were recorded.
The comparison is between two conditions: "without emitter" and "with emitter", emitter meaning presence of another group of flies in the smaller circular area of the arena floor (reward zone) prior to the test recording (see the manuscript for details).
This dataset contains processed data: pooled histograms of flies' positions and the computed fractions of time spent in the reward zone.

The displacement experiment was performed on individual flies.
The fly was walking freely in the 60 cm diameter arena and was tracked using IR sensitive camera.
We performed stimulation by the red LED,
when the fly was inside the reward zone (a small circular area of the floor).
After stimulation the fly was displaced on a flat slider.
In the non-rewarded condition the LED was plugged out, but the times when it would be turned on are indicated with the positive value of intensity as in rewarded condition.
The dataset contains the preprocessed recorded trajectories (coordinates over time), as well as the files with the processed data, necessary to generate the figures of the associated article.


## Description of the data and file structure
Pheromones folder contains files related to the pheromone experiment.
All the other files relate to the displacement experiment.

### pheromones - folder related to the pheromone experiment
#### `*.npz` files
These files contain 2d histogram values of flies' positions for two experimental conditions: with emitter and without emitter.
They are used to produce Figure 1B.
The `npz` files contain keys: `h` for the matrix of histogram values; `xe`, `ye` for the x and y edges (as the outputs of `numpy.histogram2d()` function).
* `heatmap_emitter.npz`, `heatmap_no_emitter.npz` - for time period from 5 to 30 minute;
* `heatmap_emitter1_15.npz`, `heatmap_no_emitter1_15.npz`  - for time period from 1 to 15 minute.

#### `pheromone_test_fracs5_30.tsv`
The table with metadata and calculated fractions of time spent by the group of flies inside the reward zone during each trial.
Every row corresponds to one trial.

Useful columns:
* `host` - experimental setup id
* `t` - date and time when the recording was made
* `experiment` - experiment condition; `mass_test_no` means no flies have previously been in the arena, `mass_test` means that another group of flies was kept inside the reward zone and exposed to the optogenetic reward prior the test group.
* `fly_info` - some information about the experimental flies in JSON format. `dob` - date of birth, `starve` - from what time the flies were starved on water.
* `fraction` - the relative amount of time flies spent inside the reward zone in 5-30 min of the recording.

### `temperature.csv`
The file contains temperature measurements in the fly walking arena used in the displacement experiment, contributes to Figure S2A of the paper. The arena is circular, has 60 cm diameter and the walls are heated.
The measurements were taken along two axes.

File contents:
* `axis`: x / y
* `comment`
* `t` &mdash; arena temperature readout, &deg;C
* `time` &mdash; reading time in format HH:MM
* `troom` &mdash; room temperature readout
* `x`, `y` &mdash; coordinates on the arena floor, cm, where (0, 0) is the arena center

### `flytrax20181204_170930`
This is a folder that contains files associated with one trial of the displacement experiment,
used to create figure S3A of the manuscript, which illustrates the downsampling procedure
applied to the trajectory data.
The filename contains the date and time when the recording was started (04.12.2018 at 17:09:30).
The files contain coordinates of a fly walking freely in the arena (60 cm diameter).
The coordinates were recorded in camera pixels.

Three files represent processing stages:
1) `clean_flytrax20181204_170930.csv` contains the tracking software output after manual procedure of filtering out the false detected points.

    Columns of the dataframe:
   * `time, timestamp` - self-explanatory
   * `t`, s - time from recording start
   * `frame` - frame number from the camera
   * `x_px, y_px` - coordinates of the detected object in camera view
   * `slope`, deg - not informative in this case (should represent the heading of the fly, but in this setup the fly is too small for correct estimation)
   * `led_1, led_2, led_3`, relative units - the intensity of stimulation LEDs. Only LED1 was used to deliver the stimulus (optogenetic reward, see the publication for details).
   * `central_moment, keep` - uninformative

2) `ds_t01_flytrax20181204_170930.csv` contains the data downsampled by time with step 0.1 s
   * `frame` value was averaged across the contributing frames of the "raw" recording from step 1
   * other columns as described in 1
3) `ds_t01_flytrax20181204_170930.csv` contains the data downsampled by time with step 0.1 s and by cumulative distance with step 2 px.
   * `cum_dist` is total distance traveled from the start of the recording (in pixels) with a step of 2 px.

### `all_ds_t01_cm.csv.gz`

Trajectories from all analyzed recordings, downsampled by time with step 0.1 s. Columns description same as for the file `all_ds_t01_d2_cm_no2.csv.gz`.

### `all_ds_t01_d2_cm_no2.csv.gz`

Trajectories from all analyzed recordings, downsampled by time with step 0.1 s and by distance with step 2 px.
All columns with suffix "_cm" are representing distances converted from corresponding columns without prefix in pixels to centimeters.
Columns additional to described above for file `ds_t01_flytrax20181204_170930.csv`:

* `x_cm`, `y_cm` - fly coordinates in cm
* Recording identification and metadata:
  * `fname` - filename of the original recording, contains date and time when the recording was started.
  * `condition` - rewarded/non-rewarded
  * `fly` - trial id (int)
* Experiment stage related:
  * `segment` - current experiment stage (baseline/stimulation/test_before_movement/relocation/after_relocation)
  * `tseg` - time from start of the current stage (in seconds)
  * `segment_start_ts` - timestamp when the current stage started
* `transx`/`transx_cm`, `transy`/`transy_cm` - amount of displacement along x and y axis respectively until the current moment.

* Distance traveled:
  * `cum_dist_fly` / `cum_dist_fly_cm` - total distance traveled from the recording start to the current moment
  * `cum_dist_seg` / `cum_dist_seg_cm` - distance traveled from the experiment stage start to the current moment
* Reward zone related:
  * `fr_x_px`, `fr_y_px`; `fr_x_cm`, `fr_y_cm` - fly coordinates relative to fictive reward
  * `estimated_food_x`, `estimated_food_y` (`/*_cm`) - fictive RZ coordinates after displacement / actual reward coordinates before displacement
  * `distance_reward` / `distance_reward_cm` - distance to actual RZ
  * `dist_fictive_reward` / `dist_fictive_reward_cm` - distance to fictive RZ

### Pickle files
These files are to be used for plotting of the arena border and the reward zones, they store objects of the `WalkingFlyArena` class from here https://github.com/strawlab/titova_et_al_displacement_supplemental/blob/main/arena.py.

* `big_arena_fr_black_shadow.pickle` - for the displacement experiment (arena diameter: 60 cm)
* `pheromones_mass_arena.pickle` - for the pheromone experiment (arena diameter: 20 cm)

### `stats` folder
Different values extracted or calculated from the fly trajectories in the displacement experiment.

#### `after_reloc_state.tsv`
A table representing the state right after displacement. Used for the arcs in Figure 3E, S3E
* Fly coordinates in the arena: `x_px`, `y_px` in pixels / `x_cm`, `y_cm` in meters
* Trial id: `fname` - original recording filename containing time of recording; condition: rewarded/non-rewarded
* Fictive RZ coordinates: `estimated_food_x`, `estimated_food_y` in pixels / `estimated_food_x_cm`, `estimated_food_y_cm` in cm
* Actual RZ coordinates: `reward_x_cm`, `reward_y_cm` (same for all recordings)
* Negative displacement vector: `reward_relative_x`, `reward_relative_y`, in cm
* Directional/angular information (angles relative to horizontal axis with the origin on end of displacement), all angles in radians:
  * Direction from end of displacement to the fictive RZ center: `angle_fr`
  * Direction from end of displacement to the actual RZ center: `angle_ar`
  * Direction from end of displacement to the start of displacement: `angle_reloc_start`
  * The angular size of the view from end of displacement to the actual/fictive RZ: `ar_span` / `fr_span`

#### `enter_exit_intersections.tsv`
The table containing calculated values of intersection between trajectories just before displacement and just after displacement (see Methods of the associated article for details). Plotted in Figure S3G.
* `fly` - fly id (int)
* `condition` - rewarded / non-rewarded
* `is_close` - number of points marked as overlapping
* `count_intersect` - number of points in the longest continuously intersecting segment of the trajectory
* `max_intersect_len` - length of the longest continuously intersecting segment
* `threshold` - the maximal distance between points to be marked as overlapping.

#### `enters_2cm_walking.csv`, `exits_2cm_walking.csv`
Trajectories of the flies before displacement (`enters_2cm_walking.csv`) and after displacement (`exits_2cm_walking.csv`),
cut off by 2 cm radius. The files contain trajectory data after 2-stage downsampling procedure, corresponding to walking.
The columns are as in `all_ds_t01_d2_cm_no2.csv`.
These data were used to compute the intersection score (stored in enter_exit_intersections.tsv) and to create figure S3F.

#### `fictive_reward_locations.csv`
Contains coordinates in cm of the fictive reward zone for all recordings.
* `fly` - recording id (int)
* `condition` - rewarded / non-rewarded
* `estimated_food_x_cm`, `estimated_food_y_cm` - coordinates of the fictive RZ in cm.

#### `relocation_stats_t01_no2.csv`
Table showing amount of time spent in reward zones and in the arena center during different stages.
* `at_center` - fraction of time spent in the center (a circle with radius same as RZ)
* `at_fictive_reward`, `at_reward` - fraction of time spent in fictive RZ and in actual RZ respectively
* `condition` - rewarded/non-rewarded
* `cum_dist_seg_cm` - total distance walked during this experiment stage
* `fly` - trial id (int)
* `fname` - filename of the recording (contains time when the trial was recorded)
* `segment` - stage of the experiment: baseline / stimulation / post-stim / relocation / after_relocation / test100 / test200. After_relocation includes the whole time after displacement, test100 and test200 include 100 and 200 seconds after displacement respectively.
* `tseg` - duration of the experiement stage

#### `relocation_stats_time_fractions_no2.csv`
Reshaped data from `relocation_stats_t01_no2.csv`, used for plotting (Figure 3C).
* `fly`, `fname`, `condition`, `segment` - as in `relocation_stats_t01_no2.csv`.
* `location` - location in the arena where the fraction is calculated: at_center, at_fictive_reward, at_reward.
* `fraction` - the fraction of time spent by the fly at location, specified in the specific location during specific experiment stage.

#### `start_vectors.tsv`
The computed vectors and angles describing first 5 seconds after displacement, used to create Figure 3D.
In the first 5 seconds after displacement the fly, displaced from point A to B, walks from B to F.
The fly was rewarded in a circle centered at the point R - actual reward zone.
The fictive reward zone is centered at the point R', so that AB = RR' and AB || RR'.

* `fly` - fly id (int)
* `condition` - rewarded/non-rewarded
* `anti_reloc`, \[x(cm),y(cm)\] - vector opposite to displacement vector (BA, "relocation" term was historically used instead of "displacement")
* `actual_reward_vec`, \[x(cm),y(cm)\] - BR
* `fictive_reward_vec`, \[x(cm),y(cm)\] - BR'
* `fly_vec`, \[x(cm),y(cm)\] - BF
* `angle_FRZ`, radians - angle FBR'
* `angle_ARZ`, radians - angle FBR,
* `angle_reloc`, radians - angle FBA

### `reward_zones` folder
The files associated with locations of fictive and actual RZ.
* `cmarena_rewarded_fictive_rewards` - WalkingFlyArena object with al fictive RZ indicated.
* `reward_locations.csv` - coordinates of actual reward zone, fictive reward zone and displacement vector for each trial.
  * Actual reward zone: `reward_x_cm`, `reward_y_cm`
  * Fictive reward zone: `estimated_food_x`, `estimated_food_y` (in pixels) / `estimated_food_cm`, `estimated_food_cm` (in cm)
  * `fly` - trial id
  * `condition`: rewarded/non-rewarded
  * `reward_relative_x`, `reward_relative_y` -- coordinates of actual reward zone relative to fictive RZ (negative displacement)
* `mean_fictive_rz_coords.csv` - coordinates of fictive reward zone, averaged across trials in an experimental condition (rewarded or non-rewarded).
* `mean_rewards_coords.csv` - averaged coordinates of fictive reward (`estimated_food_<x/y>_cm`),
actual reward zone coordinates (`reward_x/y_cm`, same in all trials),
average displacement (`reward_relative\_<x/y>`, in cm)

## Sharing/Access information

Links to other publicly accessible locations of the data:
  * https://github.com/strawlab/titova_et_al_displacement_supplemental
  * https://doi.org/10.5281/zenodo.7814469

## Code/Software

Software can be run with Python 3.9.
