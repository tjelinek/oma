from dataclasses import dataclass, asdict, field

import torch
from kornia.geometry.conversions import axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_from_euler
from typing import List


def default_initial_rotation():
    return torch.Tensor([0., 0., 0.])


def default_initial_translation():
    return torch.Tensor([0., 0., 0.])


@dataclass
class MovementScenario:
    steps: int = 0
    initial_rotation: torch.Tensor = field(default_factory=default_initial_rotation)
    initial_translation: torch.Tensor = field(default_factory=default_initial_translation)
    rotations: torch.Tensor = None
    translations: torch.Tensor = None

    def __post_init__(self):
        if self.rotations is None and self.translations is None:
            self.rotations = torch.zeros(1, 3)
            self.translations = torch.zeros(1, 3)

        elif self.rotations is None:
            self.rotations = torch.zeros(len(self.translations), 3)

        elif self.translations is None:
            assert self.rotations is not None
            self.translations = torch.zeros(len(self.rotations), 3)

        self.steps = len(self.translations)

    @property
    def rotation_quaternions(self) -> torch.Tensor:
        return axis_angle_to_quaternion(self.rotation_axis_angles)

    @property
    def rotation_axis_angles(self) -> torch.Tensor:
        return torch.deg2rad(self.rotations)

    def get_dict(self):
        scenario_dict = asdict(self)

        scenario_dict['rotation_quaternions'] = self.rotation_quaternions

        return scenario_dict


def get_full_rotation(step=10.0):
    return torch.arange(0., 360. + 1e-10, step)


def generate_zero_rotations(steps=72) -> MovementScenario:
    rotations = torch.zeros((steps, 3))
    return MovementScenario(rotations=rotations)


def generate_rotations_x(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations = torch.stack([rotations_x, torch.zeros_like(rotations_x), torch.zeros_like(rotations_x)], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_y(step=10.0) -> MovementScenario:
    rotations_y = get_full_rotation(step)
    rotations = torch.stack([torch.zeros_like(rotations_y), rotations_y, torch.zeros_like(rotations_y)], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_z(step=10.0) -> MovementScenario:
    rotations_z = get_full_rotation(step)
    rotations = torch.stack([torch.zeros_like(rotations_z), torch.zeros_like(rotations_z), rotations_z], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_xy(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = get_full_rotation(step)
    rotations = torch.stack([rotations_x, rotations_y, torch.zeros_like(rotations_x)], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_xz(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_z = get_full_rotation(step)
    rotations = torch.stack([rotations_x, torch.zeros_like(rotations_x), rotations_z], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_yz(step=10.0) -> MovementScenario:
    rotations_y = get_full_rotation(step)
    rotations_z = get_full_rotation(step)
    rotations = torch.stack([torch.zeros_like(rotations_y), rotations_y, rotations_z], dim=1)
    return MovementScenario(rotations=rotations)


def generate_rotations_xyz(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = get_full_rotation(step)
    rotations_z = get_full_rotation(step)
    rotations = torch.stack([rotations_x, rotations_y, rotations_z], dim=1)
    return MovementScenario(rotations=rotations)


def random_walk_on_a_hemisphere(n_steps=200, seed=42) -> MovementScenario:
    torch.manual_seed(seed)

    rotations = torch.zeros((n_steps, 3))

    steps_thresholds = torch.tensor([0.5, 0.25, 0.75])

    steps_per_ax_low = torch.tensor([0, -5, 10])
    steps_per_ax_high = torch.tensor([5, 10, 15])

    for step in range(1, n_steps):
        directions = torch.rand(3) > steps_thresholds
        steps = torch.stack(
            [torch.tensor([torch.FloatTensor(1).uniform_(low, high).item()]) for low, high in zip(steps_per_ax_low, steps_per_ax_high)]
        ).view(-1)
        steps *= directions.float()

        rotations[step] = rotations[step - 1] + steps

    rotations = rotations.deg2rad_()
    rotations = quaternion_to_axis_angle(torch.stack(quaternion_from_euler(rotations[:, 0], rotations[:, 1], rotations[:, 2]), dim=-1))
    rotations.rad2deg_()

    return MovementScenario(rotations=rotations)


def random_walk_on_a_sphere(n_steps=200, seed=42) -> MovementScenario:
    torch.manual_seed(seed)

    rotations = torch.zeros((n_steps, 3))

    steps_thresholds = torch.tensor([0.5, 0.25, 0.5])

    steps_per_ax_low = torch.tensor([0, -5, 10])
    steps_per_ax_high = torch.tensor([5, 10, 15])

    for step in range(1, n_steps):
        directions = torch.rand(3) > steps_thresholds
        steps = torch.stack(
            [torch.tensor([torch.FloatTensor(1).uniform_(low, high).item()]) for low, high in zip(steps_per_ax_low, steps_per_ax_high)]
        ).view(-1)
        steps *= directions.float()

        rotations[step] = rotations[step - 1] + steps

    rotations = rotations.deg2rad_()
    rotations = quaternion_to_axis_angle(torch.stack(quaternion_from_euler(rotations[:, 0], rotations[:, 1], rotations[:, 2]), dim=-1))
    rotations.rad2deg_()

    return MovementScenario(rotations=rotations)


def steps_to_periodic_linspace(steps):
    return torch.linspace(0, 2 * torch.pi, steps + 1)[:-1]


def generate_sinusoidal_translations(steps=72) -> MovementScenario:
    x = steps_to_periodic_linspace(steps)
    translations_x = torch.sin(x) * 0.5
    translations_y = torch.zeros_like(translations_x)
    translations_z = torch.zeros_like(translations_x)

    translations = torch.stack([translations_x, translations_y, translations_z], dim=1)
    return MovementScenario(translations=translations)


def generate_circular_translation(steps=72) -> MovementScenario:
    x = steps_to_periodic_linspace(steps)
    translations_x = torch.cos(x) * 0.5
    translations_y = torch.sin(x) * 0.5
    translations_z = torch.zeros_like(translations_x)

    translations = torch.stack([translations_x, translations_y, translations_z], dim=1)
    initial_translation = torch.tensor([0.0, 0.0, 0.0])
    return MovementScenario(translations=translations, initial_translation=initial_translation)


def generate_translation(steps: int, axes: List[str]) -> MovementScenario:
    translations = torch.zeros((steps, 3))
    for axis in axes:
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        translations[:, axis_index] = torch.arange(steps)

    return MovementScenario(translations=translations, initial_translation=torch.tensor([0.0, 0.0, 0.0]))


# Specific functions for common cases
def generate_x_translation(steps: int) -> MovementScenario:
    return generate_translation(steps, ['x'])


def generate_y_translation(steps: int) -> MovementScenario:
    return generate_translation(steps, ['y'])


def generate_z_translation(steps: int) -> MovementScenario:
    return generate_translation(steps, ['z'])


def generate_xy_translation(steps: int) -> MovementScenario:
    return generate_translation(steps, ['x', 'y'])


def generate_xyz_translation(steps: int) -> MovementScenario:
    return generate_translation(steps, ['x', 'y', 'z'])


def generate_in_depth_translations(steps=72) -> MovementScenario:
    x = steps_to_periodic_linspace(steps)
    translations_x = torch.sin(x) * 1
    translations_y = torch.zeros_like(translations_x)
    translations_z = torch.cos(x) * 4

    translations = torch.stack([translations_x, translations_y, translations_z], dim=1)
    initial_translation = torch.tensor([0.0, 0.0, -4.0])
    return MovementScenario(translations=translations, initial_translation=initial_translation)


def generate_translation_that_is_off(steps=72) -> MovementScenario:
    step = steps_to_periodic_linspace(steps)
    x = torch.linspace(0, (steps - 1) * step.item(), steps) * 0
    translations_x = torch.cos(x)
    translations_y = torch.sin(x)
    translations_z = torch.zeros_like(translations_x)

    translations = torch.stack([translations_x, translations_y, translations_z], dim=1)
    return MovementScenario(translations=translations)
