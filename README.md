# Overview of the project

[Smart Traffic light control using A2C](DL_Project___Smart_Traffic_Simulation_DRAFT.pdf)

# Note

This project was done on my University cluster (SUMO simulator and all necessary libraries were already setup)

# Sumo Simulation API

This repository serves as the main hub for maintaining the high level sumo simulation API as it pertains to RL algorithms.

## Install
```python
pip install .
```
## Install (Edit mode)
```python
pip install -e .
```
## Convention
Every API function contributed here must abide by the following conventions:
- The behavior and signature of the routine/function must be document at the opening of said API function. Below is an example of routine documents:
  ```python 
  def some_function(param_1: SomeType, param_2: SomeOtherType) -> ReturnType:
      """
      In here, we will describe what the function does.

      Parameters
      ----------
          param_1: SomeType
              This is the first parameter.
          param_2: SomeOtherType
              This is the second parameter.

      Returns
      -------
          ReturnType
              This is the return value
      """

      # Your code goes here...
  ```
- Every new API added must abide by the type hinting convention shown above.
- When commenting, it is best to categorize the comments based on its message, and the maintainer' tag. For example:
  ```python
  # NOTE(Abid): This is a normal comment that contains the tag of the maintainer who wrote it: Abid.
  # TODO(Abid): This is a to-do comment, for code that needs change.
  # WARNING(Abid): This is warning comment, used for code that is either buggy/crashes or has edge cases one must be aware.
  ```
- When dealing with classes, member names starting with `-` are to be considered "protected" and those with `--` are "private".

## Status
At the current moment, most of the functionalities implemented have been tested.
The model created by Philipp Grill and Max Ksoll are compatible with the API.
The logging functionality has been standardized and the first pass' been completed.

NOTE: There are some routines that are not up to par, most of which have been annoted with appropriate comments.

## Test Run
To run the model, clone the reposiory and run the following cli command (within the cluster):
```bash
cd examples/grill
python3 grill_eval.py --weight_loc <location to model weight> --scenario scenario1 --config_path <path to *.sumocfg file> --sumo <path to sumo home dir>
```
```bash
cd examples/frank
python3 main.py test -a TransferLight-A2C-random -s cologne3
main.py --sumo <path to sumo home dir> test -a TransferLight-A2C-random -s cologne3
```
```bash
cd examples/ksoll
python3 ksoll_eval.py --sumo <path to sumo home dir>
```
