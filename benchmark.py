import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Event, Lock, Manager
from threading import Thread
from types import SimpleNamespace

import numpy as np

from control.geometric_control import GeometricControl
from control.hinfty_control import HinftyControl
from control.lqr_control import LQR
from control.nmpc import NMPC
from control.rl_control import RLController
from models.quadrotor import QuadrotorEnv


CONTROLLER_CHOICES = ["GEOMETRIC_CTRL", "NMPC", "LQR", "HINFTY_CTRL", "RL"]


@dataclass
class BenchmarkResult:
    run_id: int
    elapsed_sec: float
    sim_time_sec: float
    steps_completed: int
    terminated: bool
    truncated: bool
    final_position_error: float
    final_velocity_error: float


def build_args(namespace: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        vehicle_cfg=namespace.vehicle_cfg,
        motion_cfg=namespace.motion_cfg,
        dt=namespace.dt,
        iterations=namespace.iterations,
        ctrl=namespace.ctrl,
        traj=namespace.traj,
        plan_yaw_traj=namespace.plan_yaw_traj,
        random_start=namespace.random_start,
        renderer="offline",
        animate="no",
        plot="no",
        model_path=namespace.model_path,
        deterministic=namespace.deterministic,
        ppo_device=namespace.ppo_device,
    )


def create_controller(args: SimpleNamespace):
    if args.ctrl == "GEOMETRIC_CTRL":
        return GeometricControl(args)
    if args.ctrl == "NMPC":
        return NMPC(args)
    if args.ctrl == "LQR":
        return LQR(args)
    if args.ctrl == "HINFTY_CTRL":
        return HinftyControl(args)
    if args.ctrl == "RL":
        return RLController(args)
    raise ValueError(f"Unknown controller: {args.ctrl}")


def run_single_simulation(run_id: int, namespace: argparse.Namespace, progress_state, progress_lock) -> BenchmarkResult:
    args = build_args(namespace)
    controller = create_controller(args)
    env = QuadrotorEnv(args, controller=controller)

    terminated = False
    truncated = False
    steps_completed = 0

    start_time = time.perf_counter()
    env.reset(seed=namespace.seed +
              run_id if namespace.seed is not None else None)
    initial_action = controller.run(env, record=False)
    env.reset_esc(initial_action)

    for _ in range(args.iterations):
        action = controller.run(env)
        _, reward, terminated, truncated, _ = env.step(action)
        steps_completed += 1
        with progress_lock:
            progress_state["steps_completed"] += 1
        if terminated or truncated:
            break

    elapsed_sec = time.perf_counter() - start_time
    sim_time_sec = steps_completed * args.dt
    position_error = float(np.linalg.norm(
        env.uav_dynamics.get_position() - env.curr_xd))
    velocity_error = float(np.linalg.norm(
        env.uav_dynamics.get_velocity() - env.curr_vd))

    return BenchmarkResult(
        run_id=run_id,
        elapsed_sec=elapsed_sec,
        sim_time_sec=sim_time_sec,
        steps_completed=steps_completed,
        terminated=terminated,
        truncated=truncated,
        final_position_error=position_error,
        final_velocity_error=velocity_error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel quadrotor benchmark runner")
    parser.add_argument("--vehicle_cfg", type=str,
                        default="quadrotor_f450.yaml")
    parser.add_argument("--motion_cfg", type=str, default="motion_normal.yaml")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--ctrl", type=str,
                        default="GEOMETRIC_CTRL", choices=CONTROLLER_CHOICES)
    parser.add_argument("--traj", type=str, default="EIGHT")
    parser.add_argument("--plan_yaw_traj", type=str, default="yes")
    parser.add_argument("--random_start", type=str, default="no")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Number of worker processes and simulations to launch")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional base random seed; each run uses seed + run_id")
    parser.add_argument("--model_path", type=str,
                        default="runs/ppo_quadrotor/best/best_model.zip")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ppo-device", dest="ppo_device",
                        type=str, default="cpu")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(completed: int, total: int, steps_completed: int, total_steps: int, start_time: float, lock: Lock) -> None:
    elapsed_sec = time.perf_counter() - start_time
    with lock:
        print(
            f"[{completed}/{total}] elapsed={format_duration(elapsed_sec)} "
            f"steps={steps_completed}/{total_steps}",
            flush=True,
        )


def heartbeat_loop(state, total: int, total_steps: int, start_time: float, interval_sec: float, lock: Lock, stop_event: Event) -> None:
    while not stop_event.wait(interval_sec):
        with lock:
            completed = state["completed"]
            steps_completed = state["steps_completed"]
        if completed >= total:
            break
        print_progress(completed, total, steps_completed,
                       total_steps, start_time, lock)


def print_summary(results: list[BenchmarkResult], total_elapsed_sec: float, workers: int) -> None:
    elapsed = np.array([item.elapsed_sec for item in results], dtype=float)
    pos_err = np.array(
        [item.final_position_error for item in results], dtype=float)
    vel_err = np.array(
        [item.final_velocity_error for item in results], dtype=float)
    steps = np.array([item.steps_completed for item in results], dtype=int)
    terminations = sum(1 for item in results if item.terminated)
    truncations = sum(1 for item in results if item.truncated)

    print(
        f"Completed {len(results)} runs with {workers} worker(s) in {total_elapsed_sec:.3f}s")
    print(f"Average wall time per run: {elapsed.mean():.3f}s")
    print(f"Average steps completed: {steps.mean():.1f}")
    print(f"Average final position error: {pos_err.mean():.6f} m")
    print(f"Average final velocity error: {vel_err.mean():.6f} m/s")
    print(f"Terminated early: {terminations}, truncated: {truncations}")
    print("Per-run details:")
    for item in sorted(results, key=lambda result: result.run_id):
        print(
            f"  run={item.run_id:03d} steps={item.steps_completed:6d} "
            f"elapsed={item.elapsed_sec:.3f}s pos_err={item.final_position_error:.6f} "
            f"vel_err={item.final_velocity_error:.6f} terminated={item.terminated} "
            f"truncated={item.truncated}"
        )


def main() -> None:
    args = parse_args()
    workers = max(1, args.workers)
    results: list[BenchmarkResult] = []
    total_steps = workers * args.iterations

    manager = Manager()
    progress_state = manager.dict(completed=0, steps_completed=0)
    print_lock = manager.Lock()
    heartbeat_stop = Event()

    total_start = time.perf_counter()
    heartbeat = Thread(target=heartbeat_loop, args=(progress_state, workers,
                       total_steps, total_start, 5.0, print_lock, heartbeat_stop), daemon=True)
    heartbeat.start()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_simulation, run_id, args,
                                   progress_state, print_lock) for run_id in range(workers)]
        for future in as_completed(futures):
            results.append(future.result())
            with print_lock:
                progress_state["completed"] += 1

    heartbeat_stop.set()
    heartbeat.join()
    print_progress(progress_state["completed"], workers,
                   progress_state["steps_completed"], total_steps, total_start, print_lock)
    total_elapsed_sec = time.perf_counter() - total_start

    print_summary(results, total_elapsed_sec=total_elapsed_sec,
                  workers=workers)


if __name__ == "__main__":
    main()
