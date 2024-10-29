# Simple model for 1D heat transfer

1D heat transfer equation used: ${dT\over dt} = α \left({d^2T\over dx^2} + {\dot{e}_{gen}\over k}\right)$

- Discretised in x and in time

- Graphs temperature as a function of x and time

## To Run Model:
- Run ```python run.py``` to run model and graph
- Configure model by editing config.toml in text editor
- Packages required: numpy, toml

  Install packages with:
  - ```python -m pip install numpy```
  - ```python -m pip install toml```
<br>
  
- Access the simulation results data in ```temperatures_C``` variable in the main code in run.py
  - For example, ```temperatures_C[3][7]``` is the 3rd time step, at the 7th discretised x distance
- Model uses x=0 as center point of plane
