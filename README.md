# MasterThesis2024

## Installation:

### Install SUMO-RL
Based on the steps described here: https://lucasalegre.github.io/sumo-rl/install/install/

 ```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

export LIBSUMO_AS_TRACI=1

conda env create --name thesis python=3.9 --file=environments.yml
pip install sumo-rl
```


# TODO's
Check impact of delta time on training
-> See notes
TODO: remove default as it does not take into account the delay caused by the yellow lights

# References
https://github.com/lweitkamp/option-critic-pytorch


# Commands used to run experiments:

### 1.1: Fine-tune reward

```
python finetune_reward.py -r intelli_light_reward
```

### 1.2: A2C training

Training: 
```
python a2c.py -t custom-2way-single-intersection-low -s 250000 -r intelli_light_reward
python a2c.py -t custom-2way-single-intersection-high -s 250000 -r intelli_light_reward
```

Evaluation: 
```
python evaluate.py -t custom-2way-single-intersection-low -m a2c_custom-2way-single-intersection-low_250000_stepsintelli_light_reward_delay_3_waiting_time_2_light_switches_1;
python evaluate.py -t custom-2way-single-intersection-low -m a2c_custom-2way-single-intersection-high_250000_stepsintelli_light_reward_delay_3_waiting_time_2_light_switches_1;
python evaluate.py -t custom-2way-single-intersection-high -m a2c_custom-2way-single-intersection-low_250000_stepsintelli_light_reward_delay_3_waiting_time_2_light_switches_1;
python evaluate.py -t custom-2way-single-intersection-high -m a2c_custom-2way-single-intersection-high_250000_stepsintelli_light_reward_delay_3_waiting_time_2_light_switches_1;
```

Investigation


### 1.3 Option-critic training

Training:
```

python option_critic_training.py -t custom-2way-single-intersection3 -r intelli_light_reward  --max_steps_total 250000 --num_options 2
python option_critic_training_curriculum.py -t custom-2way-single-intersection3 -r intelli_light_reward  --max_steps_total 250000 --num_options 2

```

Evaluation:
```
python evaluate.py -t custom-2way-single-intersection3 -m option_critic_discrete_2_options_custom-2way-single-intersection3_250000_steps;
python evaluate.py -t custom-2way-single-intersection3 -m option_critic_discrete_curriculum_2_options_custom-2way-single-intersection3_250000_steps;
```

### Visualization

Important for the visualisation is that you do not set the traci environment variable.

```
python visualize_sumo.py -t custom-2way-single-intersection3 -m option_critic_discrete_curriculum_2_options_custom-2way-single-intersection3_250000_steps
```

# Experiment 2

### Command used to generate 3x3 grid


tls.set: defines which traffic lights are controllable

Define phases as
<phase duration="33" state="GrrrrrGrrrrr"/>
<phase duration="5" state="YrrrrrYrrrrr"/>
<phase duration="33" state="rGrrrrrGrrrr"/>
<phase duration="5" state="rYrrrrrYrrrr"/>
<phase duration="33" state="rrGrrrrrGrrr"/>
<phase duration="5" state="rrYrrrrrYrrr"/>
<phase duration="33" state="rrrGrrrrrGrr"/>
<phase duration="5" state="rrrYrrrrrYrr"/>
<phase duration="33" state="rrrrGrrrrrGr"/>
<phase duration="5" state="rrrrYrrrrrYr"/>
<phase duration="33" state="rrrrrGrrrrrG"/>
<phase duration="5" state="rrrrrYrrrrrY"/>

netgenerate \
  --grid \
  --grid.number=3 \
  --grid.length=200 \
  --turn-lanes=2 \
  --turn-lanes.length=10 \
  --grid.attach-length=200 \
  --no-turnarounds=True \
  --tls.set=A0,A1,A2,B0,B1,B2,C0,C1,C2 \
  --output-file=3x3Grid3lanes.net.xml


--fringe-factor max  sets all trips to start from the edges of the network. Prevents cars from randomly teleporting inside crossroads.


python $SUMO_HOME/tools/randomTrips.py \
  -n 3x3Grid3lanes.net.xml \
  -o 3x3Grid3lanes.trips.xml \
  -b 0 -e 3600 \
  -p 5 \
  --fringe-factor max \
  --random

duarouter \
  -n 3x3Grid3lanes.net.xml \
  -t 3x3Grid3lanes.trips.xml \
  -o 3x3Grid3lanes.rou.xml \
  --ignore-errors true

### 2.1: Fine-tune reward

```
python finetune_reward.py -r intelli_light_prcol_reward
```

### 2.2: A2C Training

Training

```
python a2c.py -t 3x3grid-3lanes2 -s 250000 -r intelli_light_prcol_reward 
python a2c.py -t 3x3grid-3lanes2 -s 250000 -r intelli_light_prcol_reward -b
```

Evaluate

```
python evaluate.py -t 3x3grid-3lanes2 -m a2c_3x3grid-3lanes2_250000_stepsintelli_light_prcol_reward_delay_3_waiting_time_2_light_switches_1_out_lanes_availability_1;
python evaluate.py -t 3x3grid-3lanes2 -m a2c_broken_3x3grid-3lanes2_250000_stepsintelli_light_prcol_reward_delay_3_waiting_time_2_light_switches_1_out_lanes_availability_1;
python evaluate.py -t 3x3grid-3lanes2 -b -m a2c_3x3grid-3lanes2_250000_stepsintelli_light_prcol_reward_delay_3_waiting_time_2_light_switches_1_out_lanes_availability_1;
python evaluate.py -t 3x3grid-3lanes2 -b -m a2c_broken_3x3grid-3lanes2_250000_stepsintelli_light_prcol_reward_delay_3_waiting_time_2_light_switches_1_out_lanes_availability_1;
```


### 2.3 Option critic training


```

python option_critic_training.py -t 3x3grid-3lanes2 --broken -r intelli_light_prcol_reward  --max_steps_total 250000 --num_options 2
python option_critic_training_curriculum.py -t 3x3grid-3lanes2 --broken -r intelli_light_prcol_reward  --max_steps_total 250000 --num_options 2

```

Evaluation:
```
python evaluate.py -t 3x3grid-3lanes2 --broken --broken-mode partial -m option_critic_discrete_2_options_3x3grid-3lanes2_248400_steps;
python evaluate.py -t 3x3grid-3lanes2 --broken --broken-mode partial -m option_critic_discrete_curriculum_2_options_3x3grid-3lanes2_248400_steps;
```