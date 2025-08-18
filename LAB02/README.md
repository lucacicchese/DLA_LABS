# DLA_LABS

## Contents

1. [Completed exercises](#completed-exercises)
2. [Detailed file structure](#file-structure)
3. [Environment](#environment)
4. [Exercise 1](#exercise-1---reinforce-on-cartpole)
5. [Exercise 2](#exercise-2---reinforce--baseline-on-cartpole)
6. [Exercise 3](#exercise-31---reinforce-on-lunar-lander)

## Completed exercises

|  Exercise   | DONE  | WIP |
|-----|---|---|
| LAB01 Exercise 1 | ✅ |  |
| LAB01 Exercise 2 | ✅ |  |
| LAB01 Exercise 3.1 | ✅ |  |

## File Structure

```linux
LAB02
│   README.md
│   environment.yml
│   exercise_1.py
│   exercise_2.py
│   exercise_3.py
│   data.py
│   evaluate.py
│   models.py
│   reinforce.py
│   
└───logs
    └─── checkpoints
    └─── tensorboard
 ```

## Environment

The testing environment has been managed with anaconda, it can be created by simply running:
`conda env create -f environment.yml`

## Exercise 1 - Reinforce on cartpole

The cartpole environment is a simple environment that is usually used to  

### Implementation 1

### Results 1

## Exercise 2 - Reinforce + baseline on cartpole

To improve the basic reinforce algorithm we can add a baseline function ...

### Implementation 2

In this exercise I added a second network based on the same model as the policy that represented the value function. This model ``

### Results 2

## Exercise 3.1 - Reinforce on Lunar Lander

I was able to solve the lunar lander environment with both the basic reinforce algorithm and the one with the baseline from the previous exercises. There was of course a difference in epochs required to get to a winning state. I also noticed during training that the big difference in points assigned to different actions influences the speed at which some actions are learnt. The agent learns to avoid crashing at all costs quite fast as that gives a -100 reward but it takes a long time to learn to turn off the engins once it landed as keeping them on only gives a -0.3 or -0.03 per frame.

### Implementation 3.1

The implementation is the same used for the previous exercise but the main file can be found at `Exercise3.py`

### Results 3.1
