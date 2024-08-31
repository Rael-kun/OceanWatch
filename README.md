# OceanWatch
Anomaly detection AI using open-source AIS data 


## Usage notes

- The training data used for this project is a little too large to be conveniently managed by Git so we host it on [Kaggle](https://www.kaggle.com/datasets/nickolasweir/trainmoving) instead. 

- Loss history generated with the most up-to-date code in this repository will contain smaller values than ones showed in the graphs in the folder "Loss_history". This is due to a change in the logging behaviour since those models were trained. Examples on how to modify the current code to see similar results can be found in corresponding section in the notebook "Trajectory_transformer.ipynb".

- There is a linear layer in the TrajectoryEncoder (see "Trajectory_transformer.ipynb") that is unused. If this layer is removed it will cause issues when loading the models included in this repository as the tensors will no longer match. If training new models this will not be a problem and the line can be safely removed. 
