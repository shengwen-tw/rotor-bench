# RotorBench

A lightweight Python-based quadrotor flight simulator for benchmarking, rapid prototyping, and academic research.

## Features

1. **Lightweight quadrotor simulator**  
   A pure Python implementation designed for rapid prototyping and low-level control algorithm development.
2. **Physics engine**  
   - Accurate SE(3) rigid-body dynamics update with *RK4 integration* and *Cayley transform*.  
   - First-order delayed ESC (Electronic Speed Controller) model.  
3. **GPU-accelerated batched rollouts**  
   Built on PyTorch for massively parallel simulation.
5. **Rich controller suite out-of-the-box**  
   - Geometric Tracking Controller
   - Linear Quadratic Regulator (LQR)
   - H∞ Robust Controller
   - Nonlinear Model Predictive Control (NMPC)
   - Reinforcement Learning (PPO via Stable-Baselines3)
6. **Trajectory planning and visualization**  
   - YAML-configurable reference trajectories
   - Real-time animated 3D visualization of quadrotor motion

## Installation

```shell
pip install -r requirements.txt
```

## Usage

### Geometric Tracking Controller

```bash
python run.py --ctrl=GEOMETRIC_CTRL
```

### Linear Quadratic Regulator (LQR)

```bash
python run.py --ctrl=LQR
```

### H∞ Controller

```bash
python run.py --ctrl=HINFTY_CTRL
```

### Nonlinear Model Predictive Control (NMPC)

```bash
python run.py --ctrl=NMPC
```

### Reinforcement Learning (Experimental)

Download a pre-trained policy:
```
mkdir -p runs/ppo_quadrotor/best
wget -O runs/ppo_quadrotor/best/best_model.zip https://raw.githubusercontent.com/shengwen-tw/rotor-bench/blob/runs/ppo_quadrotor/best/best_model.zip
```

Start simulation with the RL controller:
```
python run.py --ctrl=RL
```

To train an RL policy:
```
python train_rl.py --traj HOVERING --iterations 1000 --n-envs 64 --total-steps 1000000000000
tensorboard --logdir runs/ppo_quadrotor
```

## Project Structure

```
rotor-bench/
├── run.py                   # Entry point of the simulation
├── train_rl.py              # RL training script
├── trajectory_planner.py    # Trajectory planner
├── models/
│   ├── dynamics.py          # Rigid-body dynamics
│   ├── esc.py               # ESC model
│   ├── quadrotor.py         # Gym-compatible quadrotor environment
│   ├── se3_math.py          # SE(3) math utilities
│   └── thrust_allocator.py  # Control allocation
├── control/
│   ├── care_sda.py          # CARE solver via SDA
│   ├── geometric_control.py # Geometric controller
│   ├── hinf_syn.py          # H∞ control synthesizer
│   ├── hinfty_control.py    # H∞ controller
│   ├── lqr_control.py       # LQR controller
│   ├── nmpc.py              # Nonlinear MPC
│   └── rl_control.py        # RL controller
├── configs/                 # Vehicle and trajectory configs
├── viz/                     # 3D visualization
├── assets/                  # Images and media
└── requirements.txt
```

## Preview

![](assets/preview.png)

## Citation

If you find **RotorBench** useful for your research, please consider citing:

```bibtex
@misc{rotorbench2026,
  author = {Cheng, Sheng-Wen},
  title  = {RotorBench: A Lightweight Python Quadrotor Simulator for Control Benchmarking},
  year   = {2026},
  url    = {https://github.com/shengwen-tw/rotor-bench}
}
```

An extended version of this work is currently under preparation for submission to arXiv and a peer-reviewed journal.

## License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute without restriction.
