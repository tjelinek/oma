import csv
import math
from pathlib import Path
from typing import Iterable, Dict, Tuple, cast, List

import imageio
import torch
import trimesh
from torchvision import transforms

from configs.glopose_config import RendererConfig


def load_texture(texture_path: Path, texture_size: int) -> torch.Tensor:
    texture = torch.from_numpy(imageio.v2.imread(texture_path))
    texture = texture.permute(2, 0, 1)[None].cuda() / 255.0
    if max(texture.shape[-2:]) > texture_size:
        resize = transforms.Resize(size=texture_size)
        texture = resize(texture)

    return texture


def load_mesh(mesh_path: Path):
    import kaolin
    from kaolin.io.utils import mesh_handler_naive_triangulate

    gt_mesh_prototype: kaolin.rep.SurfaceMesh = kaolin.io.obj.import_mesh(str(mesh_path), with_materials=True,
                                                                          heterogeneous_mesh_handler=mesh_handler_naive_triangulate)

    gt_mesh_prototype.uvs[:, 1] = 1.0 - gt_mesh_prototype.uvs[:, 1]
    # Fixing import that was changed in this commit
    # https://github.com/NVIDIAGameWorks/kaolin/commit/9cf895aa9af7769ae6ef23e654c1a42bcf094988

    return gt_mesh_prototype


def load_mesh_using_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    return cast(trimesh.Trimesh, mesh)


def load_gt_annotations_file(file_path) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initialize empty lists to store the data
    frames = []
    rotations_degrees = []
    translations = []

    # Load the CSV file
    with open(file_path, 'r') as csvfile:
        reader: Iterable[Dict] = csv.DictReader(csvfile)
        for row in reader:
            # Append frame number
            frames.append(int(row['frame']))

            # Append rotations in degrees
            rotations_degrees.append([
                float(row['rot_x']),
                float(row['rot_y']),
                float(row['rot_z'])
            ])

            # Append translations
            translations.append([
                float(row['trans_x']),
                float(row['trans_y']),
                float(row['trans_z'])
            ])

    # Convert rotations from degrees to radians
    rotations_radians = [[math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2])] for rot in
                         rotations_degrees]

    # Create tensors
    rotations_tensor = torch.tensor(rotations_radians).cuda()
    translations_tensor = torch.tensor(translations).cuda()

    return rotations_tensor, translations_tensor


def load_gt_data(renderer: RendererConfig):
    gt_texture = None
    gt_mesh = None
    gt_rotations = None
    gt_translations = None
    if renderer.gt_texture_path is not None:
        gt_texture = load_texture(Path(renderer.gt_texture_path), renderer.texture_size)
    if renderer.gt_mesh_path is not None:
        gt_mesh = load_mesh(Path(renderer.gt_mesh_path))
    if hasattr(renderer, 'gt_track_path') and renderer.gt_track_path is not None:
        gt_rotations, gt_translations = load_gt_annotations_file(renderer.gt_track_path)

    return gt_texture, gt_mesh, gt_rotations, gt_translations


def is_video_input(input_paths: List[Path]) -> bool:
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return len(input_paths) == 1 and input_paths[0].suffix.lower() in video_extensions


def get_scales():
    return {'m': 1.0, 'dm': 10.0, 'cm': 100.0, 'mm': 1000.0}


def get_scale_from_meter(output_scale):
    scales = get_scales()
    if output_scale not in scales:
        raise ValueError(f"Unknown unit: {output_scale}")
    return scales[output_scale]


def get_scale_to_meter(input_scale):
    return 1.0 / get_scale_from_meter(input_scale)
