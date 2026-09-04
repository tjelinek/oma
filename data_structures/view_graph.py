import pickle
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import networkx as nx
import pycolmap
import torch
import torchvision.utils as vutils
from kornia.geometry import Se3, Quaternion
from tqdm import tqdm

import adapters.cnos_adapter  # noqa: F401 — ensures cnos on sys.path for pickle deserialization


def _ensure_pathlib_compat():
    """Register a pathlib._local shim so pickles created with Python 3.13+ can be loaded on older Pythons."""
    if 'pathlib._local' not in sys.modules:
        import pathlib
        mod = types.ModuleType('pathlib._local')
        for name in dir(pathlib):
            if not name.startswith('__'):
                setattr(mod, name, getattr(pathlib, name))
        sys.modules['pathlib._local'] = mod


from data_structures.data_graph import DataGraph
from data_structures.keyframe_buffer import FrameObservation
from onboarding.colmap_utils import merge_two_databases, merge_colmap_reconstructions


@dataclass
class ViewGraphNode:
    Se3_obj2cam: Se3
    observation: FrameObservation
    colmap_db_image_id: int
    colmap_db_image_name: str
    dino_descriptor: torch.Tensor


class ViewGraph:
    def __init__(self, object_id: int | str, colmap_db_path: Path,
                 colmap_output_path: Path, device: str):
        self.view_graph = nx.DiGraph()
        self.object_id: int | str = object_id
        self.colmap_db_path: Path = colmap_db_path
        self.colmap_reconstruction_path: Path = colmap_output_path
        self.device: str = device

        # Onboarding metadata (runtime-only, not persisted in pickle by default)
        self.reconstruction_success: bool = False
        self.alignment_success: bool = False
        self.frame_filtering_time: float = 0.0
        self.matching_time: float = 0.0
        self.reconstruction_time: float = 0.0
        self.num_input_frames: int = 0
        self.colmap_num_reconstructions: int = 1
        self.image_name_to_frame_id: dict[str, int] = {}
        self.gt_model_path: Path | None = None

    def add_node(self, node_id, Se3_obj2cam, observation, colmap_db_image_id, colmap_db_image_name,
                 dino_descriptor: torch.Tensor):
        """Adds a node with ViewGraphNode attributes.

        Args:
            dino_descriptor: Pre-computed DINOv2 descriptor tensor for this node.
        """
        self.view_graph.add_node(node_id, data=ViewGraphNode(Se3_obj2cam, observation, colmap_db_image_id,
                                                             colmap_db_image_name, dino_descriptor))

    def get_node_data(self, frame_idx) -> ViewGraphNode:
        """Returns the ViewGraphNode data for a given node ID."""
        if frame_idx in self.view_graph:
            return self.view_graph.nodes[frame_idx]["data"]
        else:
            raise KeyError(f"Node {frame_idx} not found in the graph.")

    def _get_concatenated_attribute(self, attr_name: str) -> torch.Tensor:
        tensors = []
        for node_idx in sorted(self.view_graph.nodes):
            node = self.get_node_data(node_idx)
            tensors.append(getattr(node.observation, attr_name))
        return torch.cat(tensors)

    def get_concatenated_images(self) -> torch.Tensor:
        return self._get_concatenated_attribute('observed_image')

    def get_concatenated_segmentations(self) -> torch.Tensor:
        return self._get_concatenated_attribute('observed_segmentation')

    def save_viewgraph(self, save_dir: Path, colmap_reconstruction: pycolmap.Reconstruction,
                       save_images: bool = False, overwrite: bool = True, to_cpu: bool = False):
        """Saves the graph structure and associated images/segmentations to disk."""
        graph_path = save_dir / Path("graph.pkl")

        if save_dir.exists() and overwrite:
            shutil.rmtree(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        reconstruction_path = save_dir / 'reconstruction' / '0'
        reconstruction_path.mkdir(exist_ok=True, parents=True)
        print(f"[pycolmap4-debug] save_viewgraph: writing reconstruction to {reconstruction_path}")
        colmap_reconstruction.write(str(reconstruction_path))
        print(f"[pycolmap4-debug] save_viewgraph: write done")
        self.colmap_reconstruction_path = reconstruction_path

        new_db_path = save_dir / self.colmap_db_path.name
        if self.colmap_db_path != new_db_path and self.colmap_db_path.exists():
            shutil.copy(self.colmap_db_path, new_db_path)
            self.colmap_db_path = new_db_path

        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.is_file()

        if to_cpu:
            self.send_to_device('cpu')

        with open(graph_path, "wb") as f:
            pickle.dump(self, f)
            print(f"Written view graph to {str(graph_path)}")

        if save_images:
            image_save_dir = save_dir / "images"
            segmentations_save_dir = save_dir / "segmentations"

            image_save_dir.mkdir(exist_ok=True)
            segmentations_save_dir.mkdir(exist_ok=True)

            # Save images and segmentations separately
            for node_id, data in self.view_graph.nodes(data=True):
                node_data: ViewGraphNode = self.get_node_data(node_id)

                image = node_data.observation.observed_image  # Shape: (1, C, H, W)
                segmentation = node_data.observation.observed_segmentation  # Shape: (1, 1, H, W)

                img_path = image_save_dir / f"{node_id}_image.png"
                seg_path = segmentations_save_dir / f"{node_id}_seg.png"

                vutils.save_image(image, str(img_path))
                vutils.save_image(segmentation.float(), str(seg_path))

    def send_to_device(self, device):
        """Sends the graph to a given device."""
        self.device = device
        for node_id, data in self.view_graph.nodes(data=True):
            node_data: ViewGraphNode = data["data"]
            node_data.observation = node_data.observation.send_to_device(device)
            node_data.Se3_obj2cam = node_data.Se3_obj2cam.to(device)
            node_data.dino_descriptor = node_data.dino_descriptor.to(device)


def load_view_graph(load_dir: Path, device='cuda',
                    remap_paths: dict[str, str] | None = None) -> ViewGraph:
    """Loads the graph structure and associated images/segmentations from disk.

    Args:
        load_dir: Directory containing graph.pkl and reconstruction.
        device: Device to send tensors to.
        remap_paths: Optional dict of prefix substitutions to apply to stored paths
                     (e.g. {'/old/results/root/': '/new/results/root/'}), useful when a
                     cached ViewGraph is read back on a different machine.
    """
    _ensure_pathlib_compat()

    graph_path = load_dir / "graph.pkl"

    with open(graph_path, "rb") as f:
        view_graph: ViewGraph = pickle.load(f)

    if remap_paths:
        _remap_view_graph_paths(view_graph, remap_paths)

    view_graph.send_to_device(device)

    return view_graph


def _remap_view_graph_paths(view_graph: ViewGraph, remap: dict[str, str]) -> None:
    """Replace path prefixes in a ViewGraph's stored paths (colmap_db_path, colmap_reconstruction_path)."""
    for old_prefix, new_prefix in remap.items():
        db_str = str(view_graph.colmap_db_path)
        if db_str.startswith(old_prefix):
            view_graph.colmap_db_path = Path(db_str.replace(old_prefix, new_prefix, 1))
        rec_str = str(view_graph.colmap_reconstruction_path)
        if rec_str.startswith(old_prefix):
            view_graph.colmap_reconstruction_path = Path(rec_str.replace(old_prefix, new_prefix, 1))


def view_graph_from_datagraph(structure: nx.DiGraph, data_graph: DataGraph,
                              colmap_reconstruction: pycolmap.Reconstruction | None, colmap_db_path,
                              colmap_output_path, object_id: int | str) -> ViewGraph:
    """Create a ViewGraph from the keyframe graph and data graph.

    Always creates a ViewGraph with nodes for all keyframes. If colmap_reconstruction
    is provided, populates COLMAP poses and DINOv2 descriptors on the nodes.
    """
    all_image_names = [str(data_graph.get_frame_data(i).image_filename)
                       for i in range(len(data_graph.G.nodes))]

    view_graph = ViewGraph(object_id, colmap_db_path, colmap_output_path,
                           data_graph.storage_device)

    if colmap_reconstruction is not None:
        print(f"[pycolmap4-debug] view_graph_from_datagraph: reading poses from reconstruction "
              f"({colmap_reconstruction.num_images()} images)")
        from adapters.cnos_adapter import create_descriptor_extractor

        descriptor_extractor = create_descriptor_extractor()

        for image_id, image in colmap_reconstruction.images.items():
            frame_index = all_image_names.index(image.name)

            image_t_obj2cam = torch.tensor(image.cam_from_world().translation)[None]
            image_q_obj2cam_xyzw = torch.tensor(image.cam_from_world().rotation.quat)[None]
            image_q_obj2cam_wxyz = image_q_obj2cam_xyzw[:, [3, 0, 1, 2]]

            Se3_obj2cam = Se3(Quaternion(image_q_obj2cam_wxyz), image_t_obj2cam)

            frame_observation = data_graph.get_frame_data(frame_index).frame_observation

            # Compute DINOv2 descriptor for this node
            descriptor = _compute_dino_descriptor(
                frame_observation.observed_image, frame_observation.observed_segmentation, descriptor_extractor)

            view_graph.add_node(frame_index, Se3_obj2cam, frame_observation, image_id, image.name, descriptor)

        view_graph.reconstruction_success = True
    else:
        # No reconstruction — add keyframe nodes without COLMAP poses or descriptors
        for frame_index in sorted(structure.nodes()):
            frame_observation = data_graph.get_frame_data(frame_index).frame_observation
            image_name = all_image_names[frame_index]
            Se3_identity = Se3.identity(1)
            view_graph.view_graph.add_node(
                frame_index,
                data=ViewGraphNode(Se3_identity, frame_observation, -1, image_name, torch.empty(0))
            )
        view_graph.reconstruction_success = False

    return view_graph


def _compute_dino_descriptor(image_tensor: torch.Tensor, segmentation_mask: torch.Tensor,
                             descriptor_extractor) -> torch.Tensor:
    """Compute a DINOv2 descriptor from an image tensor and segmentation mask."""
    from einops import rearrange
    from torchvision.ops import masks_to_boxes

    segmentation_mask = segmentation_mask.squeeze(0)
    if segmentation_mask.sum() == 0:
        # Mask collapse (e.g. SAM2 lost the object): no bbox to crop a descriptor from.
        # Empty descriptor matches the convention of nodes without reconstruction.
        return torch.empty(0)
    segmentation_bbox = masks_to_boxes(segmentation_mask)
    image_np = rearrange((image_tensor * 255).to(torch.uint8), '1 c h w -> h w c').numpy(force=True)
    _cls_descriptor, dense_descriptor = descriptor_extractor.extract_descriptors(
        image_np, segmentation_mask, segmentation_bbox)
    return dense_descriptor.squeeze()


def compute_dino_descriptors_for_view_graph(view_graph: ViewGraph, dino_model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DINOv2 descriptors for all nodes in a ViewGraph from saved image files.

    This is a standalone function — the ViewGraph itself has no dependency on the descriptor model.

    Args:
        view_graph: The ViewGraph whose nodes need descriptors.
        dino_model: A CustomDINOv2 model instance (from descriptor_from_hydra()).

    Returns:
        Tuple of (cls_descriptors, dense_descriptors) tensors, one row per node
        in sorted node order.
    """
    cls_descriptors = []
    dense_descriptors = []

    viewgraph_save_path = view_graph.colmap_db_path.parent
    img_save_dir = viewgraph_save_path / 'images'
    seg_save_dir = viewgraph_save_path / 'segmentations'

    for node_id in sorted(view_graph.view_graph.nodes):
        node = view_graph.get_node_data(node_id)

        colmap_db_img_name = node.colmap_db_image_name
        img_name = f'{Path(colmap_db_img_name).stem}_image.png'
        seg_name = f'{Path(colmap_db_img_name).stem}_seg.png'

        img_path = img_save_dir / img_name
        seg_path = seg_save_dir / seg_name

        dino_cls_descriptor, dino_dense_descriptor = dino_model.get_detections_from_files(img_path, seg_path)

        cls_descriptors.append(dino_cls_descriptor)
        dense_descriptors.append(dino_dense_descriptor)

    cls_descriptors = torch.cat(cls_descriptors)
    dense_descriptors = torch.cat(dense_descriptors)

    return cls_descriptors, dense_descriptors


def merge_two_view_graphs(viewgraph1_folder: Path, viewgraph2_folder: Path, merged_folder: Path,
                          use_icp: bool = True,
                          onboarding_config: 'OnboardingConfig | None' = None,
                          device: str = 'cuda') \
        -> tuple['ViewGraph', 'pycolmap.Reconstruction', Dict[str, str], Dict[str, str]]:
    """Merge two ViewGraphs (e.g. down + up) into one.

    Args:
        viewgraph1_folder: Path to first (target/fixed) ViewGraph cache folder.
        viewgraph2_folder: Path to second (source) ViewGraph cache folder.
        merged_folder: Output path for the merged ViewGraph.
        use_icp: If True, align rec2 → rec1 before merging.
        onboarding_config: If provided, use matching-based alignment (preferred).
            Falls back to ICP-only when None.
        device: PyTorch device string for matching.

    Returns:
        merged_viewgraph: The merged ViewGraph object.
        merged_reconstruction: The merged pycolmap.Reconstruction.
        db1_image_rename_dict: {original_name -> prefixed_name} for viewgraph1.
        db2_image_rename_dict: {original_name -> prefixed_name} for viewgraph2.
    """
    if merged_folder.exists():
        shutil.rmtree(merged_folder)
    merged_folder.mkdir(parents=True, exist_ok=True)

    view_graph1 = load_view_graph(viewgraph1_folder, device='cpu')
    view_graph2 = load_view_graph(viewgraph2_folder, device='cpu')

    colmap_db1_path = view_graph1.colmap_db_path
    colmap_db2_path = view_graph2.colmap_db_path

    merged_db_path: Path = merged_folder / "database.db"
    db1_image_rename_dict, db2_image_rename_dict = merge_two_databases(colmap_db1_path, colmap_db2_path, merged_db_path)

    merged_db = pycolmap.Database.open(str(merged_db_path))

    viewgraph1_node_relabel_mapping, vg1_colmap_id_map = relabel_viewgraph_nodes(merged_db, view_graph1,
                                                                                 db1_image_rename_dict)
    viewgraph2_node_relabel_mapping, vg2_colmap_id_map = relabel_viewgraph_nodes(merged_db, view_graph2,
                                                                                 db2_image_rename_dict)

    copy_relabeled_images(viewgraph1_folder, viewgraph1_node_relabel_mapping, merged_folder)
    copy_relabeled_images(viewgraph2_folder, viewgraph2_node_relabel_mapping, merged_folder)

    merged_reconstruction_path = merged_folder / 'reconstruction'

    reconstruction1 = pycolmap.Reconstruction(str(view_graph1.colmap_reconstruction_path))
    reconstruction2 = pycolmap.Reconstruction(str(view_graph2.colmap_reconstruction_path))

    align_info = {}
    if use_icp and onboarding_config is not None:
        # Matching-based alignment (uses dense/sparse matcher to find 3D correspondences)
        from data_providers.flow_provider import create_matching_provider
        from onboarding.colmap_utils import align_reconstructions_matching
        match_provider = create_matching_provider(
            onboarding_config.filter_matcher, onboarding_config, device)
        reconstruction2, align_info = align_reconstructions_matching(
            reconstruction1, reconstruction2, match_provider,
            target_images_dir=viewgraph1_folder / 'images',
            source_images_dir=viewgraph2_folder / 'images',
            target_segs_dir=viewgraph1_folder / 'segmentations',
            source_segs_dir=viewgraph2_folder / 'segmentations',
            sample_size=onboarding_config.sample_size,
            certainty_threshold=onboarding_config.min_certainty_threshold,
            reliability_threshold=onboarding_config.flow_reliability_threshold,
            black_background=onboarding_config.merge_black_background,
            use_procrustes=onboarding_config.merge_use_procrustes,
            refine_with_icp=onboarding_config.merge_refine_with_icp,
            icp_centroid_prewarp=onboarding_config.merge_icp_centroid_prewarp,
            device=device)
    elif use_icp:
        from onboarding.colmap_utils import align_reconstructions_icp
        reconstruction2, align_info = align_reconstructions_icp(reconstruction1, reconstruction2)

    merged_reconstruction = merge_colmap_reconstructions(
        reconstruction1, reconstruction2,
        vg1_colmap_id_map, vg2_colmap_id_map,
        db1_image_rename_dict, db2_image_rename_dict,
    )

    merged_viewgraph = ViewGraph(view_graph1.object_id, merged_db_path, merged_reconstruction_path, view_graph1.device)

    merged_viewgraph_G = nx.compose(view_graph1.view_graph, view_graph2.view_graph)
    merged_viewgraph.view_graph = merged_viewgraph_G
    merged_viewgraph.save_viewgraph(merged_folder, merged_reconstruction, save_images=True, overwrite=False,
                                    to_cpu=True)

    return merged_viewgraph, merged_reconstruction, db1_image_rename_dict, db2_image_rename_dict, align_info


def copy_relabeled_images(source_viewgraph_folder, viewgraph_node_relabel_mapping, target_viewgraph_folder):
    viewgraph_img_folder = source_viewgraph_folder / 'images'
    viewgraph_seg_folder = source_viewgraph_folder / 'segmentations'
    merged_img_folder = target_viewgraph_folder / 'images'
    merged_seg_folder = target_viewgraph_folder / 'segmentations'

    merged_img_folder.mkdir(parents=True, exist_ok=True)
    merged_seg_folder.mkdir(parents=True, exist_ok=True)

    for old_img_id, new_img_id in viewgraph_node_relabel_mapping.items():
        old_image_path = viewgraph_img_folder / f'{old_img_id}_image.png'
        old_seg_path = viewgraph_seg_folder / f'{old_img_id}_seg.png'

        new_image_path = merged_img_folder / f'{new_img_id}_image.png'
        new_seg_path = merged_seg_folder / f'{new_img_id}_seg.png'

        shutil.copy(old_image_path, new_image_path)
        shutil.copy(old_seg_path, new_seg_path)


def relabel_viewgraph_nodes(merged_db: pycolmap.Database, view_graph: ViewGraph,
                            db_image_rename_dict: Dict[str, str] = None) -> tuple[Dict[Any, Any], Dict[int, int]]:
    """Relabel viewgraph nodes to match merged DB image IDs and names.

    Returns:
        viewgraph_node_relabel_mapping: {old_node_id: new_colmap_image_id}
        colmap_id_mapping: {old_colmap_image_id: new_colmap_image_id}
    """
    all_merged_images = {image.name: image for image in merged_db.read_all_images()}
    viewgraph_node_relabel_mapping = {}
    colmap_id_mapping = {}
    for node_id in view_graph.view_graph.nodes:
        node = view_graph.get_node_data(node_id)
        old_colmap_id = node.colmap_db_image_id
        old_image_name = node.colmap_db_image_name
        new_image_name = db_image_rename_dict[old_image_name]

        merged_db_image = all_merged_images[new_image_name]
        new_image_colmap_id = merged_db_image.image_id

        node.colmap_db_image_id = new_image_colmap_id
        node.colmap_db_image_name = new_image_name

        viewgraph_node_relabel_mapping[node_id] = new_image_colmap_id
        colmap_id_mapping[old_colmap_id] = new_image_colmap_id

    view_graph.view_graph = nx.relabel_nodes(view_graph.view_graph, viewgraph_node_relabel_mapping)

    return viewgraph_node_relabel_mapping, colmap_id_mapping


def load_view_graphs_by_object_id(view_graph_save_paths: Path, onboarding_type: str, device,
                                  remap_paths: dict[str, str] | None = None) -> Dict[Any, ViewGraph]:
    view_graphs: Dict[Any, ViewGraph] = {}
    total_dirs = sum(1 for d in view_graph_save_paths.iterdir() if d.is_dir())
    for i, view_graph_dir in tqdm(enumerate(view_graph_save_paths.iterdir()), total=total_dirs,
                                  desc="Loading view graphs"):
        if view_graph_dir.is_dir():
            if onboarding_type == 'onboarding_static':
                if not view_graph_dir.stem.endswith('_both'):
                    continue
            elif onboarding_type == 'onboarding_static_merged':
                if not view_graph_dir.stem.endswith('_merged'):
                    continue
            elif onboarding_type == 'onboarding_dynamic':
                if not view_graph_dir.stem.endswith('_dynamic'):
                    continue
            elif onboarding_type is None:
                pass
            else:
                pass
                # raise ValueError(f"Unknown onboarding type {onboarding_type}")

            view_graph: ViewGraph = load_view_graph(view_graph_dir, device=device,
                                                    remap_paths=remap_paths)
            view_graphs[view_graph.object_id] = view_graph

    return view_graphs


def export_viewgraphs_to_cnos_format(viewgraphs_home: Path, bop_home: Path, experiment: str, split: str):
    """Export ViewGraph images/segmentations to CNOS matchability_images format.

    Creates the directory structure expected by CNOS for template-based detection:
        {bop_home}/{dataset}/matchability_images_{split}/{obj_id}/rgb/
        {bop_home}/{dataset}/matchability_images_{split}/{obj_id}/mask_visib/

    Args:
        viewgraphs_home: Root of the ViewGraph cache (e.g. cache/view_graph_cache/)
        bop_home: Root of the BOP data directory (e.g. data/bop/)
        experiment: Experiment name (e.g., 'ufm_c0975r05')
        split: Onboarding split suffix to filter (e.g., 'both', 'up', 'down', 'dynamic')
    """
    path_to_experiment = viewgraphs_home / experiment

    for dataset_path in tqdm(list(path_to_experiment.iterdir()), desc="Datasets"):
        dataset = dataset_path.name
        destination_folder = bop_home / dataset / f'matchability_images_{split}'
        shutil.rmtree(str(destination_folder), ignore_errors=True)

        for view_graph_path in tqdm(list(dataset_path.iterdir()), desc=f"Sequences in {dataset}", leave=False):
            if dataset in ['hope', 'handal'] and view_graph_path.name.split('_')[-1] != split:
                continue

            sequence_name = '_'.join(view_graph_path.name.split('_')[:-1])

            view_graph_img_path = view_graph_path / 'images'
            view_graph_seg_path = view_graph_path / 'segmentations'

            destination_sequence = destination_folder / f'{sequence_name}'
            destination_images = destination_sequence / 'rgb'
            destination_segmentations = destination_sequence / 'mask_visib'

            destination_images.mkdir(parents=True, exist_ok=True)
            destination_segmentations.mkdir(parents=True, exist_ok=True)

            for file_path in view_graph_img_path.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, destination_images)

            for file_path in view_graph_seg_path.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, destination_segmentations)
