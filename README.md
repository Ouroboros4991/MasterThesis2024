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

### Install CityFlow:
Based on the steps described here: https://cityflow.readthedocs.io/en/latest/install.html#install
I install it using the source files.
```bash
sudo apt update && sudo apt install -y build-essential cmake
git clone https://github.com/cityflow-project/CityFlow.git
pip install .
conda install -c conda-forge libstdcxx-ng=12
```



# TODO's
Check impact of delta time on training
-> See notes
TODO: remove default as it does not take into account the delay caused by the yellow lights

# References
https://github.com/lweitkamp/option-critic-pytorch 