# robosuite simulator over a unix socket, built from the env_args of a robomimic hdf5 with the arm swapped by --robot.
#   python eval/robosuite_sim_server.py --env-args-h5 <robomimic hdf5> --robot Panda --address /tmp/sim.sock
# ("reset", seed) -> obs; ("step", action[7]) -> (obs, reward, success); ("info", None) -> dict; ("close", None)
import argparse
import json
from multiprocessing.connection import Listener

import h5py
import numpy as np

CAMERAS = ["agentview", "robot0_eye_in_hand"]


def make_env(env_args, robot, size):
    import robosuite
    kw = dict(env_args["env_kwargs"])
    kw["robots"] = robot
    kw.update(has_renderer=False, has_offscreen_renderer=True, ignore_done=True, use_object_obs=True, use_camera_obs=True,
              camera_depths=False, camera_names=CAMERAS, camera_heights=size, camera_widths=size)
    env = robosuite.make(env_args["env_name"], **kw)
    for name in env.observation_names:
        if "joint_pos" in name or "eef_vel" in name:
            env.modify_observable(observable_name=name, attribute="active", modifier=True)
    return env


def pack(obs, env):
    env.robots[0].controller.update(force=True)
    return dict(eef_site_mat=np.asarray(env.robots[0].controller.ee_ori_mat, np.float32),
                image=np.ascontiguousarray(obs["agentview_image"][::-1]), wrist=np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
                eef_pos=obs["robot0_eef_pos"].astype(np.float32), eef_quat_xyzw=obs["robot0_eef_quat"].astype(np.float32),
                gripper_qpos=obs["robot0_gripper_qpos"].astype(np.float32), object=obs["object-state"].astype(np.float32))


def fixtures(env):
    """Static goal fixtures of the task (not object poses): target bin quadrant centre, peg positions."""
    out = {}
    if hasattr(env, "target_bin_placements"):
        out["target_bin"] = np.asarray(env.target_bin_placements[env.object_id]).tolist()
        out["bin_size"] = np.asarray(env.bin_size).tolist()
    if hasattr(env, "peg1_body_id"):
        out["peg1"] = np.asarray(env.sim.data.body_xpos[env.peg1_body_id]).tolist()
        out["peg2"] = np.asarray(env.sim.data.body_xpos[env.peg2_body_id]).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-args-h5", required=True, help="robomimic hdf5 whose data.attrs['env_args'] define the task")
    ap.add_argument("--robot", required=True, help="robosuite robot name: Panda, IIWA, UR5e, Kinova3")
    ap.add_argument("--address", required=True, help="unix socket path")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    with h5py.File(args.env_args_h5, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    env = make_env(env_args, args.robot, args.size)
    with Listener(args.address, family="AF_UNIX") as listener:
        print("READY", flush=True)
        with listener.accept() as conn:
            while True:
                cmd, payload = conn.recv()
                if cmd == "reset":
                    np.random.seed(int(payload))
                    obs = env.reset()
                    conn.send(pack(obs, env))
                elif cmd == "step":
                    obs, reward, done, info = env.step(np.asarray(payload, dtype=np.float64))
                    conn.send((pack(obs, env), float(reward), bool(env._check_success())))
                elif cmd == "info":
                    conn.send(dict(env_name=env_args["env_name"], robot=args.robot, horizon=int(env.horizon),
                                   control_freq=float(env.control_freq), fixtures=fixtures(env),
                                   gripper=env.robots[0].gripper.__class__.__name__))
                elif cmd == "close":
                    conn.send(True)
                    break
    env.close()


if __name__ == "__main__":
    main()
