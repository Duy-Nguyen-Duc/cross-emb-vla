# MimicGen env interfaces / task configs for the robomimic tasks MimicGen does not ship: Lift (grasp + lift the cube)
# and PickPlaceCan (grasp the can, place it in the bin); Square uses MimicGen's MG_Square. Imported by mimicgen_run.py.
from mimicgen.configs.config import MG_Config
from mimicgen.env_interfaces.robosuite import RobosuiteInterface, MG_PickPlace  # noqa: F401  (registers MG_PickPlace)


class MG_Lift(RobosuiteInterface):
    """robosuite Lift: a single subtask relative to the cube."""

    def get_object_poses(self):
        return dict(cube=self.get_object_pose(obj_name=self.env.cube.root_body, obj_type="body"))

    def get_subtask_term_signals(self):
        # MimicGen expects at least one signal in datagen_info; the single subtask itself ends with the episode
        return dict(grasp=int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube)))


class Lift_Config(MG_Config):
    NAME = "lift"
    TYPE = "robosuite"

    def task_config(self):
        self.task.task_spec.subtask_1 = dict(
            object_ref="cube",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class PickPlaceCan_Config(MG_Config):
    NAME = "pick_place_can"
    TYPE = "robosuite"

    def task_config(self):
        self.task.task_spec.subtask_1 = dict(
            object_ref="can",
            subtask_term_signal="grasp_can",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()
