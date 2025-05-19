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