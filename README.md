# Assignment 2: Bayesian Network Representation and Inference

## Overview

This assignment focuses on implementing and analyzing Bayesian Networks (BNs) for probabilistic queries in a healthcare domain. The project involves working with a BN designed for a hospital's referral center for newborn babies with congenital heart disease, aiding in preliminary diagnosis and early treatment decisions.

## Prerequisites

The assignment uses several key components:

- Modules from the AI textbook (Poole and Mackworth)
- Starter code and skeleton implementation
- JSON files containing BN specifications

## Project Structure

### Provided Files

- `hw.py`: Main code skeleton
- `hw_example.py`: Example implementation using the "earthquake" BN
- `/child/`: Directory containing healthcare domain BN JSON files
- `/earthquake/`: Directory containing example BN JSON files

## Assignment Tasks

### Part 1: Exact Inference

#### Requirements

1. **BN Implementation**

- Load BN from JSON files
- Construct BeliefNetwork class instance

1. **Exact Inference Implementation**

- Implement `perform_exact_inference` function
- Use Variable Elimination algorithm
- Compute: P(Disease | CO2Report = 1, XrayReport = 0, Age = 0)

#### Analysis Tasks

- Compare two variable ordering approaches:
   1. Alphabetical ordering
   1. Optimized ordering
- Performance measurement:
  - Use timeit module
  - Average over 10 runs
  - Save results in `part1.csv`

### Part 2: Approximate Inference

#### Requirements

1. **Implementation**

- Implement `perform_approximate_inference` function
- Use Rejection Sampling algorithm

1. **Testing Scenarios**

- n_samples = 10
- n_samples = 100
- PAC-guaranteed n_samples (ε = 0.01, δ = 0.05)

#### Analysis Tasks

1. **Performance Analysis**

- Measure average runtime (10 runs)
- Save results in `part21.csv`

1. **Error Analysis**

- Calculate MSE compared to exact inference
- Run each method 10 times
- Save results in `part22.csv`

## Submission Requirements

### Required Files

1. **Code Implementation**

- `hw.py`

1. **Results**

- `part1.csv`: Exact inference timing results
- `part21.csv`: Approximate inference timing results
- `part22.csv`: Error analysis results

1. **Documentation**

- PDF report including:
  - Code explanation
  - Methodology description
  - Results analysis
  - Performance speculations

## Reference

Spiegelhalter, David J., et al. "Bayesian analysis in expert systems." Statistical Science (1993): 219-247.


<br>
