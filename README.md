# Project by Lorenzo Menchini and Christian Petruzzella

## Setup and Installation

1.  **Clone the Repository**

2.  ```
    pip install -r requirements.txt
    cd final_models/
    ```

## Test scripts for:

## Model with starting speed at 0
### 1. `env_testing.py`
* **Location**: `our_model/env_testing.py`

## Model with randomized starting speed
### 2. `env_testing_v_rand.py`
* **Location**: `our_model_varying_speed/env_testing_v_rand.py`

## Model with augmented physical characteristics
### 3. `env_testing_physical.py`
* **Location**: `original_physical_model/env_testing_physical.py`

## How to Run the Scripts

The scripts are run from the command line and accept a numerical argument that specifies the test scenario.

### Command Syntax:

First, navigate to the appropriate directory:

* For `env_testing.py`:
    ```bash
    cd final_models/our_model/
    ```
* For `env_testing_v_rand.py`:
    ```bash
    cd final_models/our_model_varying_speed/
    ```
* For `env_testing_physical.py`:
    ```bash
    cd final_models/original_physical_model/
    ```

Then, run the script:
```bash
python <script_name.py> <env_type>
```

<env_type> is a number between 1 and 4:
* 1: Easy Scenario: Jaywalker stationary, obstacle (car) distant.
* 2: Difficult Scenario: Jaywalker stationary, obstacle (car) close to the jaywalker.
* 3: Very Difficult Scenario: Jaywalker stationary, two obstacle cars in the lane.
* 4: Scenario Iteration: Sequentially runs scenarios 1, 2, 3, cyclically for the number of test episodes.
