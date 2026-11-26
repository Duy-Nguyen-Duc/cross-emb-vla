# Wrist and agentview camera calibration of the robosuite arms (env robosuite).
#   PYTHONPATH=. python data/process/robomimic/wrist_camera_calib.py [--out assets/wrist_camera/wrist_cameras_rs.json]
import argparse
import json
import math
import os

import mujoco
import numpy as np

ROBOTS = {"rs_panda": "Panda", "rs_iiwa": "IIWA", "rs_ur5e": "UR5e", "rs_kinova3": "Kinova3"}
MJ_TO_CV = np.diag([1.0, -1.0, -1.0])          # MuJoCo camera (-z forward, y up) -> OpenCV (z forward, y down)


def pose(xpos, xmat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(xmat).reshape(3, 3)
    T[:3, 3] = xpos
    return T


def cam_cv(m, d, cam):
    T = pose(d.cam_xpos[cam], d.cam_xmat[cam])
    T[:3, :3] = T[:3, :3] @ MJ_TO_CV
    return T


def intrinsics(fovy_deg, w, h):
    f = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/wrist_camera/wrist_cameras_rs.json")
    ap.add_argument("--frames", default="runs/calib_frames", help="where to save the orientation-check PNGs")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    import imageio.v2 as imageio
    import robosuite
    from robosuite.controllers import load_controller_config
    os.makedirs(args.frames, exist_ok=True)
    out = {}
    for key, name in ROBOTS.items():
        env = robosuite.make("Lift", robots=name, controller_configs=load_controller_config(default_controller="OSC_POSE"),
                             has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                             camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=args.size,
                             camera_widths=args.size, control_freq=20, ignore_done=True)
        obs = env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_eye_in_hand")
        agv = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        # tcp frame as robosuite reports it: robot0_eef_pos = grip site position, robot0_eef_quat = right_hand BODY orientation
        Ts = []
        rng = np.random.default_rng(0)
        for k in range(6):                                 # T_tcp_cam must be constant under motion (rigid mount)
            obs, *_ = env.step(np.r_[rng.uniform(-0.5, 0.5, 6), -1.0])
            T_ws = pose(d.site_xpos[site], d.xmat[hand])
            Ts.append(np.linalg.inv(T_ws) @ cam_cv(m, d, cam))
            if k == 0:
                eef_err = float(np.linalg.norm(obs["robot0_eef_pos"] - d.site_xpos[site]))
                q = obs["robot0_eef_quat"]; q_xyzw = np.r_[d.xquat[hand][1:], d.xquat[hand][0]]
                quat_err = float(min(np.linalg.norm(q - q_xyzw), np.linalg.norm(q + q_xyzw)))
        T_tcp_cam = np.mean(Ts, 0)
        spread = float(max(np.abs(T - T_tcp_cam).max() for T in Ts))
        T_w_agv = cam_cv(m, d, agv)
        out[key] = dict(robot_uid=name, gripper=env.robots[0].gripper.__class__.__name__, fov=math.radians(float(m.cam_fovy[cam])),
                        width=args.size, height=args.size, intrinsic_cv=intrinsics(float(m.cam_fovy[cam]), args.size, args.size),
                        T_tcp_cam_cv=T_tcp_cam.round(6).tolist(), T_tcp_cam_spread=spread,
                        tcp_pose_source="robot0_eef_pos (grip site) + robot0_eef_quat (right_hand body, xyzw -> wxyz)",
                        eef_site_mismatch=eef_err, eef_quat_mismatch=quat_err,
                        agentview=dict(fov=math.radians(float(m.cam_fovy[agv])), intrinsic_cv=intrinsics(float(m.cam_fovy[agv]), args.size, args.size),
                                       T_world_cam_cv=T_w_agv.round(6).tolist()),
                        note="camera frame = OpenCV (x right, y down, z forward); Lift table top at world z = 0.8")
        for cname in ("agentview", "robot0_eye_in_hand"):
            img = obs[f"{cname}_image"]
            imageio.imwrite(f"{args.frames}/{key}_{cname}_raw.png", img)
            imageio.imwrite(f"{args.frames}/{key}_{cname}_flipped.png", img[::-1])
        env.close()
        print(f"{key}: fovy {m.cam_fovy[cam]:.1f} deg, t_tcp_cam {T_tcp_cam[:3, 3].round(4).tolist()}, "
              f"rigid-mount spread {spread:.2e}, eef pos mismatch {eef_err:.2e}, eef quat mismatch {quat_err:.2e}")
    out["agentview_by_task"] = {}
    for task in ("Lift", "PickPlaceCan", "NutAssemblySquare"):      # each arena places its agentview camera differently
        env = robosuite.make(task, robots="Panda", controller_configs=load_controller_config(default_controller="OSC_POSE"),
                             has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        agv = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        out["agentview_by_task"][task] = dict(fov=math.radians(float(m.cam_fovy[agv])), intrinsic_cv=intrinsics(float(m.cam_fovy[agv]), args.size, args.size),
                                              T_world_cam_cv=cam_cv(m, d, agv).round(6).tolist(), pos=d.cam_xpos[agv].round(4).tolist())
        env.close()
        print(f"agentview {task}: pos {d.cam_xpos[agv].round(3).tolist()} fovy {m.cam_fovy[agv]:.1f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out, "and frames in", args.frames)


if __name__ == "__main__":
    main()
