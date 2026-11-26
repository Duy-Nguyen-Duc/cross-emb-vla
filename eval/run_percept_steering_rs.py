# robomimic benchmark: policy here, robosuite simulator in a separate process (eval/robosuite_sim_server.py).
#   python eval/run_percept_steering_rs.py --ckpt <stage-3 checkpoint> --name <run> --out eval_results/<run> [--steer 0|1.0]
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SIM_PYTHON = os.environ.get('SIM_PYTHON', str(Path(sys.executable).parents[2] / 'robosuite/bin/python'))
CALIB = REPO / 'assets/wrist_camera/wrist_cameras_rs.json'

ROBOTS = {'rs_panda': 'Panda', 'rs_iiwa': 'IIWA', 'rs_ur5e': 'UR5e', 'rs_kinova3': 'Kinova3'}
TASKS = {'Lift': 'Lift the cube.',
         'PickPlaceCan': 'Pick up the can and place it in the bin.',
         'NutAssemblySquare': 'Pick up the square nut and put it on the square peg.'}
ENV_ARGS_H5 = {'Lift': 'demo_data/robomimic/source/robomimic/lift_ph_low_dim.hdf5', 'PickPlaceCan': 'demo_data/robomimic/source/robomimic/can_ph_low_dim.hdf5',
               'NutAssemblySquare': 'demo_data/robomimic/source/robomimic/square_ph_low_dim.hdf5'}
PAIRS = [('rs_panda', t) for t in TASKS]                       # K-shot target robot
ALL_PAIRS = [(r, t) for r in ROBOTS for t in TASKS]
SEED_BASE, POLICY_SEED, EXEC_HORIZON, HORIZON = 2025, 7, 16, 400

STEER_GAIN = 0.011             # tcp metres per unit OSC_POSE position command per 20 Hz step (fit on the PH demos: 0.010-0.012)
STEER_RADIUS, STEER_STEPS, STEER_CLIP = 0.30, 16, 0.3   # correction spread over the whole chunk, <= 0.3 command units per step (no OSC saturation / overshoot)
TARGETS = {   # colour rule, geometry (m): settled resting height of the body origin, top-face height, shape model
    'Lift': {'cube': dict(color='red', half=0.021, rest_z=0.821, z_top=0.842, shape='cube', hold=(0.3, 0.65))},   # hold: normalised opening band when holding it
    'PickPlaceCan': {'can': dict(color='red', half=0.025, rest_z=0.860, z_top=0.900, z_hit=0.880, shape='cylinder', hold=(0.35, 0.7))},
    'NutAssemblySquare': {'nut': dict(color='brass', half=0.01, rest_z=0.830, z_top=0.840, shape='nut',
                                      tip_x=0.079, grasp_x=0.062, hold=(0.1, 0.4))},   # handle along the nut's +x: tip at 0.079, demos grasp at 0.062
}
PLACE_TASKS = {'PickPlaceCan': dict(min_z=1.0, stop=0.03)}   # place servo only above the bin walls; the square peg needs +-3 mm, beyond this mechanism
HANDOVER_DZ, HANDOVER_XY = 0.06, 0.02   # approach steering stops this high above the object top / this close in xy: the final descent is the policy's
INTERACTIONS = {   # grasp target, then (holding) the static place fixture reported by the simulator
    'Lift': dict(grasp='cube', place=None, mode='grasp'),
    'PickPlaceCan': dict(grasp='can', place='target_bin', mode='grasp', approach=False),   # approach steering cost 7/50 on the Panda (grasp quality); the place servo toward the bin quadrant is kept
    'NutAssemblySquare': dict(grasp='nut', place='peg1', mode='none'),   # all policy failures are insertion failures (+-3 mm); approach steering only hurt
}
MIN_PIXELS = 12


# ----------------------------------------------------------------------------------------------------------------
# perception
# ----------------------------------------------------------------------------------------------------------------
def quat_wxyz_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(np.int32) for i in range(3)]
    if color == 'red':
        return (r > 90) & (r > 2 * g) & (r > 2 * b)
    if color == 'brass':
        return (r > 110) & (g > 70) & (r > 1.15 * g) & (g > 1.3 * b)
    raise ValueError(color)


def largest_component(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


class Camera:
    def __init__(self, K, T_wc):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.Kinv = np.linalg.inv(self.K)
        T = np.asarray(T_wc, float).reshape(4, 4)
        self.R, self.o = T[:3, :3], T[:3, 3]

    def ray(self, u, v):
        return self.R @ (self.Kinv @ np.array([u, v, 1.0]))

    def plane_hit(self, d, z):
        if d[2] > -1e-6:
            return None
        lam = (z - self.o[2]) / d[2]
        return float(lam) if lam > 0 else None


class ObjectLocator:
    """World position of a target from the wrist camera (calibrated mount x eef pose) or the fixed agentview camera."""

    def __init__(self, robot, task, cams=('wrist', 'third')):
        self.cams, self.targets = tuple(cams), TARGETS[task]
        c = json.load(open(CALIB))[robot]
        self.K_wrist, self.T_tcp_cam = np.asarray(c['intrinsic_cv'], float), np.asarray(c['T_tcp_cam_cv'], float)
        a = json.load(open(CALIB))['agentview_by_task'][task]
        self.third = Camera(a['intrinsic_cv'], a['T_world_cam_cv'])
        self.memory = {}

    def reset(self):
        self.memory.clear()

    def views(self, obs):
        T_wt = np.eye(4)
        T_wt[:3, :3] = quat_wxyz_to_matrix(obs['tcp_pose'][3:7])
        T_wt[:3, 3] = obs['tcp_pose'][:3]
        out = []
        for name in self.cams:
            if name == 'wrist':
                out.append((name, obs['wrist'], Camera(self.K_wrist, T_wt @ self.T_tcp_cam)))
            else:
                out.append((name, obs['image'], self.third))
        return out

    def _detect(self, rgb, spec):
        mask = color_mask(rgb, spec['color'])
        if mask.sum() < MIN_PIXELS:
            return None
        mask = largest_component(mask)
        if mask.sum() < MIN_PIXELS:
            return None
        vs, us = np.nonzero(mask)
        return float(np.median(us)) + 0.5, float(np.median(vs)) + 0.5, mask, 1.0

    def _lift(self, rgb, cam, det, spec):
        """Detection (u, v, mask) -> world position of the steering target (object centre; for the nut its handle grasp point)."""
        u, v, mask, _ = det
        vs, us = np.nonzero(mask)
        shape = spec['shape']
        if shape == 'cube':                                  # top face = brightest, uniformly lit part of the silhouette
            b = rgb[mask].astype(np.int32).sum(1)
            p5, p95 = np.percentile(b, [5, 95])
            face = np.zeros_like(mask)
            face[vs, us] = b >= p95 - 0.12 * (p95 - p5)
            face = largest_component(face)
            if face.sum() >= 8:
                fv, fu = np.nonzero(face)
                d_f = cam.ray(float(np.median(fu)) + 0.5, float(np.median(fv)) + 0.5)
                lam = cam.plane_hit(d_f, spec['z_top'])
                if lam is not None:
                    p = cam.o + lam * d_f
                    p[2] = spec['rest_z']
                    return p, dict(model='topface', n_face=int(face.sum()))
            d = cam.ray(u, v)
            n = d / np.linalg.norm(d)
            lam = cam.plane_hit(d, spec['rest_z'] + spec['half'] * abs(n[2]))
            if lam is None:
                return None, {}
            p = cam.o + lam * d
            p[2] = spec['rest_z']
            return p, dict(model='centroid')
        if shape == 'cylinder':                              # silhouette centroid ray meets the plane z_hit (fit on the demos)
            d = cam.ray(u, v)
            lam = cam.plane_hit(d, spec['z_hit'])
            if lam is None:
                return None, {}
            p = cam.o + lam * d
            p[2] = spec['rest_z']
            return p, dict(model='centroid')
        # nut: lift the silhouette to the top-face plane, principal axis = handle direction (the narrow end), grasp point from the tip
        pix = np.stack([us + 0.5, vs + 0.5, np.ones(len(us))], -1)
        D = pix @ cam.Kinv.T @ cam.R.T
        ok = D[:, 2] < -1e-6
        lam = (spec['z_top'] - cam.o[2]) / np.where(ok, D[:, 2], -1.0)
        pts = (cam.o[None, :2] + lam[:, None] * D[:, :2])[ok & (lam > 0)]
        if len(pts) < 20:
            return None, {}
        c = pts.mean(0)
        w, V = np.linalg.eigh((pts - c).T @ (pts - c))
        a = V[:, int(np.argmax(w))]
        proj = (pts - c) @ a
        perp = (pts - c) @ np.array([-a[1], a[0]])
        hi, lo = proj > 0.6 * proj.max(), proj < 0.6 * proj.min()
        if hi.sum() < 3 or lo.sum() < 3:
            return None, {}
        if perp[hi].std() > perp[lo].std():                  # the handle end is the narrow one
            a = -a
            proj = -proj
        tip = c + a * float(np.percentile(proj, 98))
        grasp = tip - a * (spec['tip_x'] - spec['grasp_x'])
        return np.r_[grasp, spec['rest_z']], dict(model='nut_handle', axis=a.round(3).tolist(), n=int(len(pts)))

    def locate(self, obs, target, near=0.10):
        spec = self.targets[target]
        tcp = obs['tcp_pose'][:3]
        cands = {}
        for name, rgb, cam in self.views(obs):
            det = self._detect(rgb, spec)
            if det is None:
                continue
            pos, extra = self._lift(rgb, cam, det, spec)
            if pos is None:
                continue
            m = det[2]
            border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            cands[name] = dict(pos=pos, source=name, px=(round(det[0], 1), round(det[1], 1)), n_px=int(m.sum()), border=border, **extra)
        pick = None
        w = cands.get('wrist')
        if w is not None and w['n_px'] >= 40 and np.linalg.norm((w['pos'] - tcp)[:2]) < near and not (w['border'] and 'third' in cands):
            pick = w
        elif 'third' in cands:
            pick = cands['third']
        elif 'wrist' in cands:
            pick = cands['wrist']
        if pick is not None:
            self.memory[target] = pick['pos']
            return pick
        if target in self.memory:
            return dict(pos=self.memory[target], source='memory')
        return dict(pos=None, source='none')


def gripper_holding(state_now, closed_cmd, lo=0.08, hi=0.85):   # per-object bands in TARGETS[...]['hold']
    """Holding without privileged information: closing was commanded and the normalised opening settled strictly between
    fully closed (nothing between the fingers: 0) and open; nut handle ~0.2, cube ~0.5, can ~0.55."""
    return bool(closed_cmd) and lo < state_now < hi


class Steering:
    """Chunk-endpoint servo (as eval/run_percept_steering.py): approach target while the gripper is open, the static
    place fixture while holding; the chunk's planned xy endpoint is shifted by k * (r - planned) over its first steps."""

    def __init__(self, robot, task, k, fixtures):
        self.robot, self.task, self.k = robot, task, k
        self.spec = INTERACTIONS[task]
        self.loc = ObjectLocator(robot, task)
        self.fixtures = fixtures
        self.state, self.last, self.hist = {}, {}, []

    def reset(self):
        self.state.clear()
        self.hist.clear()
        self.loc.reset()

    def __call__(self, pos, g, obs, grip_state):
        self.last = {}
        name, place = self.spec['grasp'], self.spec['place']
        st = self.state
        st.setdefault('grip_init', float(g[0, 0]))
        held = gripper_holding(grip_state, st.get('closed_cmd', False), *self.loc.targets[name].get('hold', (0.08, 0.85)))
        st['closed_cmd'] = bool(np.any(np.abs(g[:, 0] - st['grip_init']) >= 0.25))
        tcp = obs['tcp_pose'][:3]
        if self.spec['mode'] == 'none':
            self.last.update(target=name, phase='off', held=held)
            return pos
        pspec = PLACE_TASKS.get(self.task)
        placing = bool(pspec and place and held)
        if placing:
            r = np.asarray(self.fixtures[place], float)[:3] - tcp
            self.last.update(target=place, phase='place', held=held, source='fixture')
            if tcp[2] < pspec['min_z'] or float(np.linalg.norm(r[:2])) < pspec['stop']:
                self.last['r'] = r.round(4).tolist()
                return pos
        else:
            if not self.spec.get('approach', True):
                self.last.update(target=name, phase='approach-off', held=held)
                return pos
            est = self.loc.locate(obs, name)
            self.last.update(target=name, phase='approach', held=held, source=est['source'])
            if est['pos'] is None:
                return pos
            self.hist = (self.hist + [np.asarray(est['pos'], float)])[-3:]   # static target: median of the last 3 estimates rejects a wrong detection
            r = np.median(np.stack(self.hist), 0) - tcp
            spec = self.loc.targets[name]
            if tcp[2] - spec['z_top'] < HANDOVER_DZ or float(np.linalg.norm(r[:2])) < HANDOVER_XY:
                self.last.update(r=r.round(4).tolist(), handover=True)
                return pos
        self.last['r'] = r.round(4).tolist()
        if self.k == 0:
            return pos
        open_mask = np.abs(g[:, 0] - st['grip_init']) < 0.25
        if placing:
            open_mask = ~open_mask
        rxy = r[:2]
        if not open_mask[0] or float(np.linalg.norm(rxy)) > STEER_RADIUS:
            return pos
        H = EXEC_HORIZON
        planned = STEER_GAIN * pos[:H, :2].sum(0)
        corr = self.k * (rxy - planned)
        n = min(STEER_STEPS, H)
        per_step = np.clip(corr / (n * STEER_GAIN), -STEER_CLIP, STEER_CLIP)
        pos = pos.copy()
        for t in range(n):
            if open_mask[t]:
                pos[t, :2] += per_step
        self.last.update(planned_xy=planned.round(4).tolist(), corr_xy=corr.round(4).tolist(),
                         per_step=per_step.round(4).tolist(), steered_steps=int(open_mask[:n].sum()))
        return pos


# ----------------------------------------------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------------------------------------------
def request_seed(policy_seed, robot, task, episode_seed, replan):
    key = json.dumps([policy_seed, robot, task, episode_seed, replan], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2 ** 63 - 1)


class Sim:
    """robosuite in a separate process, one per worker."""

    def __init__(self, robot, task, sock):
        env = dict(os.environ, MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', PYTHONNOUSERSITE='1',
                   LD_LIBRARY_PATH=str(Path(SIM_PYTHON).parents[1] / 'lib'))   # only the env's libs: the container path carries another torch
        if os.path.exists(sock):
            os.remove(sock)
        self.proc = subprocess.Popen([SIM_PYTHON, str(REPO / 'eval/robosuite_sim_server.py'), '--env-args-h5', str(REPO / ENV_ARGS_H5[task]),
                                      '--robot', ROBOTS[robot], '--address', sock], env=env, cwd=REPO)
        while not os.path.exists(sock):
            if self.proc.poll() is not None:
                raise RuntimeError('simulator process exited during start-up')
            time.sleep(0.2)
        self.conn = Client(sock, family='AF_UNIX')
        self.info = self.call('info', None)

    def call(self, cmd, payload):
        self.conn.send((cmd, payload))
        return self.conn.recv()

    def close(self):
        try:
            self.call('close', None)
        finally:
            self.proc.wait(timeout=60)


def run_pair(args):
    import torch
    from experiment.inference_wrapper import InferenceWrapper
    from data.process.robomimic.to_maniskill_h5 import GRIPPERS, gripper_opening
    robot, task = args.pair
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{robot}__{task}.json'
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite completed pair: {target}')
    episode_file, trace_file = out / f'{robot}__{task}.episodes.jsonl', out / f'{robot}__{task}.trace.jsonl'
    if episode_file.exists() or trace_file.exists():
        raise RuntimeError('Partial pair exists; inspect before resuming in a fresh directory')
    manifest = json.loads(Path(args.manifest).read_text())
    grange = json.loads((REPO / 'demo_data/robomimic/lerobot' / robot / 'meta/gripper_range.json').read_text())
    g_lo, g_hi = float(grange['lo']), float(grange['hi'])
    policy = InferenceWrapper(Path(args.checkpoint), device='cuda:0')
    policy.set_robot_uid(f'{robot}_rel')
    head = policy.model.flow_matching_head
    assert head.action_horizon == EXEC_HORIZON and head.num_inference_timesteps == 4, 'benchmark protocol: 4 denoising steps'
    sim = Sim(robot, task, str(Path(args.sock_dir) / f'{robot}__{task}.sock'))
    steer = Steering(robot, task, args.steer, sim.info['fixtures']) if args.steer > 0 else None

    def observe(o):
        q = o['eef_quat_xyzw']
        tcp = np.concatenate([o['eef_pos'], [q[3], q[0], q[1], q[2]]]).astype(np.float32)
        grip = float(np.clip((gripper_opening(o['gripper_qpos'][None], GRIPPERS[robot])[0, 0] - g_lo) / max(g_hi - g_lo, 1e-8), 0.0, 1.0))
        return dict(image=o['image'], wrist=o['wrist'], tcp_pose=tcp, grip=grip, site_mat=o.get('eef_site_mat'))

    def predict(element, seed):
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            return policy.get_action(element)

    def executable(chunk, obs):
        g = np.asarray(chunk['action.gripper']).reshape(EXEC_HORIZON, -1)
        pos = np.asarray(chunk['action.position'])
        if steer is not None:
            pos = steer(np.asarray(pos, dtype=float), g, obs, obs['grip'])
        return np.concatenate([pos, np.asarray(chunk['action.rotation']), g], axis=1)[:EXEC_HORIZON]

    episodes, replay_checked = [], False
    total_start = time.perf_counter()
    try:
        with episode_file.open('x') as journal, trace_file.open('x') as traces:
            for episode_index in range(args.episodes):
                seed = SEED_BASE + episode_index
                start = time.perf_counter()
                if steer is not None:
                    steer.reset()
                o = sim.call('reset', seed)
                obs = observe(o)
                infer_s = step_s = 0.0
                success, replan, actions = False, 0, []
                for step in range(args.step_limit or HORIZON):
                    if not actions:
                        state = np.concatenate([obs['tcp_pose'], [obs['grip']]]).astype(np.float32)
                        element = {'video.image': obs['image'][None], 'video.wrist_image': obs['wrist'][None],
                                   'state.position': state[None, :3], 'state.rotation': state[None, 3:7], 'state.gripper': state[None, 7:8],
                                   'annotation.human.task_description': [TASKS[task]]}
                        rs = request_seed(POLICY_SEED, robot, task, seed, replan)
                        tic = time.perf_counter()
                        pred = executable(predict(element, rs), obs)
                        infer_s += time.perf_counter() - tic
                        if not replay_checked:
                            if not np.array_equal(pred, executable(predict(element, rs), obs)):
                                raise RuntimeError('Per-request RNG replay is not exact')
                            replay_checked = True
                        traces.write(json.dumps(dict(episode_seed=seed, step=step, replan=replan, request_seed=rs, state=state.tolist(),
                                                     action=pred.tolist(), steer=dict(steer.last) if steer is not None else None)) + '\n')
                        traces.flush()
                        actions = list(pred)
                        replan += 1
                    tic = time.perf_counter()
                    o, reward, success = sim.call('step', actions.pop(0))
                    step_s += time.perf_counter() - tic
                    obs = observe(o)
                    if success:
                        break
                row = dict(episode=episode_index, episode_seed=seed, policy_seed=POLICY_SEED, success=bool(success), steps=step + 1,
                           replans=replan, seconds=time.perf_counter() - start, inference_seconds=infer_s, env_step_seconds=step_s)
                episodes.append(row)
                journal.write(json.dumps(row) + '\n')
                journal.flush()
                print(json.dumps(row), flush=True)
    finally:
        sim.close()
    n_succ = sum(x['success'] for x in episodes)
    result = dict(robot=robot, task=task, success_rate=n_succ / len(episodes), successes=n_succ, num_per_task=len(episodes),
                  policy_seed=POLICY_SEED, seed_base=SEED_BASE, request_seed_scheme='sha256-v1',
                  checkpoint_sha256=manifest['checkpoint_sha256'], replan_steps=EXEC_HORIZON, denoising_steps=4,
                  step_budget=args.step_limit or HORIZON, steering=steering_spec(args.steer), simulator=sim.info,
                  total_seconds=time.perf_counter() - total_start, replay_exact=True)
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(target)
    print(json.dumps(result, indent=2), flush=True)


def steering_spec(k):
    if k <= 0:
        return None
    return dict(source='percept', detector='color', depth='plane', cams='wrist,third', k=k, radius=STEER_RADIUS, steps=STEER_STEPS,
                place=PLACE_TASKS, handover=[HANDOVER_DZ, HANDOVER_XY], gain=STEER_GAIN, clip=STEER_CLIP, calib=CALIB.name,
                off_tasks=[t for t, v in INTERACTIONS.items() if v['mode'] == 'none'])


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stage(name, ckpt, out):
    staged = out / 'staged' / name
    staged.mkdir(parents=True, exist_ok=True)
    (staged / 'config.json').write_text(json.dumps(json.loads((ckpt.parent / 'config.json').read_text()), indent=2) + '\n')
    link = staged / ckpt.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt.resolve())
    expcfg = staged / 'experiment_cfg'
    if not expcfg.exists():
        expcfg.symlink_to(ckpt.parent / 'experiment_cfg', target_is_directory=True)
    return link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--name')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--steer', type=float, default=1.0, help='steering gain k; 0 = policy alone')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--pairs', nargs='+', type=int, default=None, help='indices into the pair list (default: all)')
    ap.add_argument('--all-pairs', action='store_true', help='all robots x tasks instead of the target robot only')
    ap.add_argument('--targets', default=None, help='K-shot target pairs for the summary split, e.g. rs_panda:Lift,rs_iiwa:PickPlaceCan '
                                                   '(default: the LPS protocol, rs_panda on every task)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--wait', type=int, default=300)
    ap.add_argument('--step-limit', type=int, default=None, help='smoke tests only')
    ap.add_argument('--pair', nargs=2, metavar=('ROBOT', 'TASK'), help='worker mode')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--sock-dir', type=Path)
    args = ap.parse_args()
    if args.pair:
        return run_pair(args)
    if not (args.ckpt and args.name and args.out):
        ap.error('--ckpt, --name and --out are required')
    out = args.out if args.out.is_absolute() else REPO / args.out
    ckpt = args.ckpt if args.ckpt.is_absolute() else REPO / args.ckpt
    plist = ALL_PAIRS if args.all_pairs else PAIRS
    pairs = [plist[i] for i in (args.pairs if args.pairs is not None else range(len(plist)))]
    sock_dir = Path(f'/dev/shm/rs_sock_{os.getpid()}')      # AF_UNIX paths are limited to ~100 characters
    sock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONPATH=str(REPO))
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(ckpt)
    staged = stage(args.name, ckpt, out)
    manifest = out / f'{args.name}_manifest.json'
    manifest.write_text(json.dumps({'checkpoint': str(staged), 'source_checkpoint': str(ckpt), 'checkpoint_sha256': digest,
                                    'experiment': args.name,
                                    'execution_protocol': {'replan_steps': EXEC_HORIZON, 'denoising_steps': 4, 'seed_base': SEED_BASE,
                                                           'policy_seed': POLICY_SEED, 'horizon': HORIZON, 'steering': steering_spec(args.steer)}},
                                   indent=2) + '\n')
    dest = out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)

    def run(job):
        robot, task = job
        if (dest / f'{robot}__{task}.json').exists():
            return f'{robot}/{task}: already done'
        cmd = [sys.executable, str(Path(__file__).resolve()), '--pair', robot, task, '--checkpoint', str(staged), '--manifest', str(manifest),
               '--out', str(dest), '--episodes', str(args.episodes), '--steer', str(args.steer), '--sock-dir', str(sock_dir)]
        if args.step_limit:
            cmd += ['--step-limit', str(args.step_limit)]
        t0 = time.time()
        with (out / 'logs' / f'{args.name}__{robot}__{task}.log').open('a') as f:
            r = subprocess.run(cmd, env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        return f'{robot}/{task}: rc={r.returncode} {(time.time() - t0) / 60:.1f} min'

    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        pending = [p for p in pairs if not (dest / f'{p[0]}__{p[1]}.json').exists()]
        if not pending:
            break
        if attempt > 1:
            part = out / f'partial_{attempt - 1}'
            part.mkdir(exist_ok=True)
            for robot, task in pending:
                for f in dest.glob(f'{robot}__{task}.*'):
                    shutil.move(str(f), part / f.name)
            print(f'{len(pending)} pair(s) unfinished; retry {attempt} in {args.wait} s', flush=True)
            time.sleep(args.wait)
        with ThreadPoolExecutor(args.workers) as ex:
            for line in ex.map(run, pending):
                print(line, flush=True)
    targets = set(PAIRS) if not args.targets else {tuple(x.split(':')) for x in args.targets.split(',')}
    rows = []
    for robot, task in pairs:
        p = dest / f'{robot}__{task}.json'
        if not p.exists():
            raise SystemExit(f'missing {p}')
        r = json.loads(p.read_text())
        rows.append(dict(robot=robot, task=task, episodes=r['num_per_task'], successes=r['successes'],
                         split='target' if (robot, task) in targets else 'source'))
    summary = dict(name=args.name, checkpoint=str(ckpt), sha256=digest, episodes=sum(r['episodes'] for r in rows),
                   successes=sum(r['successes'] for r in rows), rows=rows, minutes=round((time.time() - t0) / 60, 1),
                   steering=steering_spec(args.steer))
    for split in ('target', 'source'):
        sub = [r for r in rows if r['split'] == split]
        if sub:
            summary[f'{split}_episodes'] = sum(r['episodes'] for r in sub)
            summary[f'{split}_successes'] = sum(r['successes'] for r in sub)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    per_task = ', '.join('%s/%s=%d' % (r['robot'], r['task'], r['successes']) for r in rows)
    print(f"{args.name}: {summary['successes']}/{summary['episodes']} ({per_task}) in {summary['minutes']} min")


if __name__ == '__main__':
    sys.exit(main())
