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



# Command used to generate 3x3 grid


python $SUMO_HOME/tools/randomTrips.py \
  -n 3x3Grid2lanes.net.xml \
  -o 3x3Grid2lanes.trips.xml \
  -b 0 -e 3600 \
  -p 2 \
  --random

duarouter \
  -n 3x3Grid2lanes.net.xml \
  -t 3x3Grid2lanes.trips.xml \
  -o 3x3Grid2lanes.rou.xml