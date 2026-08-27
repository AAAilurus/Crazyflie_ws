# Crazyflie Lighthouse LQR and Inverse Reinforcement Learning

**Author:** Basanta Adhikari  
 (UTA)

This repository contains the working Crazyflie real-flight control
and inverse reinforcement learning framework developed for
Lighthouse-based experiments.

The system uses:

- Lighthouse position feedback
- outer-loop LQR position control
- inner-loop PID attitude control
- real-flight expert-data collection
- SPSA inverse LQR / inverse reinforcement learning
- learned-gain validation
- sequential expert-versus-learned Crazyflie experiments
- automatic convergence and trajectory plotting

## Repository layout

    cflib_python/
    |
    |-- UTA_lighthouse.py
    |-- UTA_lighthouse_irl_hover21.py
    |-- UTA_lighthouse_leader_dual_exact.py
    |-- UTA_lighthouse_follower_dual_182749.py
    |-- run_two_cf_182749_SEQUENTIAL.py
    |-- run_complete_irl_experiment.py
    |
    |-- erau_original_pid_lighthouse_200hz_WORKING_BASELINE.py
    |-- lqr_lighthouse_200hz_from_working_baseline.py
    |
    |-- UTA_LQR/
    |   |-- UTA_controller.py
    |   |-- UTA_outer_lqr.py
    |   `-- UTA_outer_lqr_gazebo.py
    |
    |-- controller/
    |
    |-- controller_lqr_real_200hz_FINAL_STABLE_1M/
    |
    |-- IRL_real/
    |   |-- common/
    |   |-- training/
    |   |-- learning/
    |   |-- demo/
    |   |-- WORKING_ONLY/
    |   |-- data/
    |   `-- results/
    |
    |-- training_data/
    |
    |-- plot_dual_demo_182749.py
    `-- plot_recovered_k_from_spsa.py

    docs/
    |-- SETUP_GUIDE.md
    |-- ARCHITECTURE.md
    `-- SAFETY.md

## Controller

The translational error state is

    e = [ex, ey, ez, evx, evy, evz]^T

and the outer-loop LQR policy is

    u = -K e

with

    u = [roll_des, pitch_des, az_cmd]^T.

The expert cost used in the current experiments is

    Q* = diag(8, 8, 12, 1.2, 1.2, 10)

    R* = diag(6, 6, 2)

The outer-loop LQR replaces the position PID.

The inner attitude PID remains responsible for attitude
stabilization.

## One-time setup

Read:

    docs/SETUP_GUIDE.md
    docs/ARCHITECTURE.md
    docs/SAFETY.md

Install Python dependencies:

    cd cflib_python

    python3 -m venv .venv
    source .venv/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt

Optional plotting environment:

    python3 -m venv plot_env

    plot_env/bin/pip install \
        numpy matplotlib

## Verify code before flight

    cd cflib_python

    python3 -m compileall -q .

## 1. Run the working PID baseline

Replace the radio URI with the URI of your vehicle.

    python3 erau_original_pid_lighthouse_200hz_WORKING_BASELINE.py \
        --uri <CRAZYFLIE_URI>

## 2. Run the working LQR

    python3 lqr_lighthouse_200hz_from_working_baseline.py \
        --uri <CRAZYFLIE_URI>

## 3. Collect expert IRL training data

The current training collector is:

    UTA_lighthouse_irl_hover21.py

The training phase records only the useful hover experiment.

Current 21-second experiment:

    0-5 s
        nominal hover

    5-6 s
        +0.10 m X
        +0.10 m Z

    6-13 s
        recovery / nominal hover

    13-14 s
        -0.10 m Y
        +0.10 m Z

    14-21 s
        recovery / nominal hover

Small multisine action excitation is used during the valid training
window to provide sufficient excitation for inverse learning.

Example:

    python3 UTA_lighthouse_irl_hover21.py \
        --uri <EXPERT_URI> \
        --x 0.0 \
        --y 0.0 \
        --z 1.0 \
        --yaw 0.0 \
        --takeoff-seconds 4 \
        --run-seconds 21 \
        --land-z 0.05 \
        --land-seconds 6 \
        --loop-hz 200

The generated CSV is written under:

    training_data/

## 4. Run SPSA inverse LQR

The primary implementation is:

    IRL_real/learning/spsa_matlab_version.py

Example:

    python3 IRL_real/learning/spsa_matlab_version.py \
        --csv training_data/<PROCESSED_DATA>.csv \
        --alphaQ 1e-2 \
        --c_spsa 1e-4 \
        --tol_K 1e-3 \
        --maxIter 10000 \
        --pd_floor 1e-6 \
        --reg_huu 1e-8 \
        --q0 10.0 \
        --seed 42 \
        --result-dir IRL_real/results/<RUN_NAME>

The implementation checks the Bellman-feature matrix.

For the full experiment the desired condition is

    rank(Phi) = 45 / 45

The SPSA result directory contains:

    Q_learned.csv
    K_learned.csv
    K_star.csv
    history.csv
    summary.csv
    convergence.png

## 5. Learned follower

The learned follower is:

    UTA_lighthouse_follower_dual_182749.py

The filename retains the original validated experiment identifier
for compatibility.

The learned K loaded by this script is replaced automatically by
the complete IRL pipeline after a successful SPSA run.

## 6. Expert-versus-learned flight

The current validated experiment is sequential rather than
simultaneous.

The expert leader completely lands before the learned follower
takes off.

Run:

    python3 run_two_cf_182749_SEQUENTIAL.py \
        --leader-uri <EXPERT_URI> \
        --follower-uri <FOLLOWER_URI> \
        --z 1.0 \
        --takeoff-seconds 4 \
        --run-seconds 5 \
        --land-z 0.05 \
        --land-seconds 6 \
        --loop-hz 200 \
        --gap-seconds 3

Sequence:

    Expert leader
        takeoff
        hover
        land
        motors off

    wait

    Learned follower
        takeoff
        hover
        land
        motors off

No training action excitation is used during validation.

## 7. Generate expert-versus-learned plots

    source plot_env/bin/activate

    python3 plot_dual_demo_182749.py

Generated figures include:

    Z_Leader_vs_Follower.png
    XY_Leader_vs_Follower.png

The SPSA result directory also contains the K-convergence plot.

## 8. Complete automated IRL experiment

The automation wrapper connects the complete process:

    expert training
          |
          v
    training CSV
          |
          v
    processing
          |
          v
    SPSA
          |
          v
    Q_learned + K_learned
          |
          v
    learned follower
          |
          v
    expert validation flight
          |
          v
    learned validation flight
          |
          v
    plots

Run:

    python3 run_complete_irl_experiment.py \
        --leader-uri <EXPERT_URI> \
        --follower-uri <FOLLOWER_URI> \
        --z 1.0

The pipeline checks the identification result before allowing the
learned controller to be used for the validation flight.

## IRL implementations

The repository also contains additional inverse-learning
implementations under:

    IRL_real/learning/

including offline SPSA, Z-only experiments, full symmetric-Q SPSA,
and alternative experimental versions.

The current primary real-flight method is:

    spsa_matlab_version.py

## Generated files

Flight CSVs, logs, plots, learned runtime files, virtual
environments, and Python caches are intentionally excluded from Git.

This keeps the repository focused on reproducible source code.

## Attribution

Crazyflie control components in this project build on the Bitcraze,
USC-CATT, and ERAU Crazyflie software ecosystem.

Original attribution and license notices in upstream source files
must be retained.

The UTA LQR, Lighthouse integration, IRL workflow, SPSA experiments,
automation, and experimental integration in this repository are
maintained by:

Basanta Adhikari  
The University of Texas at Arlington (UTA)
