#!/bin/bash

# module purge
# module load cuda/12.4 intel/2023 mpi cmake

# python -m experiment.train --config-name=config_flow model.version=QHNet_flow_v2 
python -m experiment.train_v2 --config-name=config_flow model.version=QHNet_flow_v3
