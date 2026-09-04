
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

from configs.components.bop_config import BaseBOPConfig
from configs.components.frame_provider_config import BaseFrameProviderConfig
from configs.matching.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching.sift_configs.base_sift_config import BaseSiftConfig
from configs.matching.tracking_configs.base_tracking_config import BaseTrackingConfig
from configs.matching.ufm_configs.base_ufm_config import BaseUFMConfig


# Roots for data, results and caches. Override them without editing this file by
# setting OMA_DATA_ROOT / OMA_RESULTS_ROOT / OMA_CACHE_ROOT, or point the individual
# dataset fields below wherever your copies live.
_DATA_ROOT = Path(os.environ.get('OMA_DATA_ROOT', 'data')).expanduser()
_RESULTS_ROOT = Path(os.environ.get('OMA_RESULTS_ROOT', 'results')).expanduser()
_CACHE_ROOT = Path(os.environ.get('OMA_CACHE_ROOT', 'cache')).expanduser()


@dataclass
class PathsConfig:
    results_folder: Path = _RESULTS_ROOT
    cache_folder: Path = _CACHE_ROOT
    purge_cache: bool = False

    bop_data_folder: Path = _DATA_ROOT / 'bop'
    ho3d_data_folder: Path = _DATA_ROOT / 'HO3D'
    ycbineoat_data_folder: Path = _DATA_ROOT / 'YCBInEOAT'
    navi_data_folder: Path = _DATA_ROOT / 'NAVI' / 'navi_v1.5'
    behave_data_folder: Path = _DATA_ROOT / 'BEHAVE'
    tum_rgbd_data_folder: Path = _DATA_ROOT / 'tum_rgbd'
    google_scanned_objects_data_folder: Path = _DATA_ROOT / 'GoogleScannedObjects'
    handal_data_folder: Path = _DATA_ROOT / 'HANDAL' 


@dataclass
class RunConfig:
    device: str = 'cuda'
    dataset: str = None
    sequence: str = None
    experiment_name: str = None
    special_hash: str = ''
    object_id: Union[int, str] = None
    # Correspondence sampling (FlowMatchingProvider.sample, inherited from RoMa) draws
    # with torch.multinomial, so an unseeded run picks a different match subset every
    # time. That propagates into keyframe selection: two runs of the SAME config were
    # observed to disagree on 25 of 39 validation cells, once by 84 keyframes vs 18.
    # Seeding per sequence makes a run reproducible and lets an ablation attribute a
    # difference to the factor it changed rather than to the sampler.
    seed: int = 0


@dataclass
class InputConfig:
    input_frames: int = None
    skip_indices: int = 1
    frame_provider: str = 'precomputed'
    frame_provider_config: BaseFrameProviderConfig = field(default_factory=BaseFrameProviderConfig)
    segmentation_provider: str = 'SAM2'
    gt_flow_source: str = 'FlowNetwork'
    image_downsample: float = 1.0
    depth_scale_to_meter: float = 1.0
    run_only_on_frames_with_known_pose: bool = True
    hot3d_device: str = 'aria'  # 'aria' or 'quest3'


@dataclass
class RANSACConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__
        self.method: str = 'pycolmap'          # 'pycolmap' | 'magsac++' | 'ransac' | '8point' | 'pygcransac'
        self.max_error: float = 0.5            # Inlier threshold (pixels)
        self.confidence: float = 0.9999        # RANSAC success probability
        self.min_num_matches: int = 5          # Skip RANSAC below this
        self.min_inlier_ratio: float = 0.25     # Floor: reliability=0 if inlier_ratio < this
        self.use_prosac: bool = False           # pygcransac only: confidence-weighted sampling
        self.min_iters: int = 1000             # pygcransac only: minimum iterations


@dataclass
class OnboardingConfig:
    # Frame filter settings
    frame_filter: str = 'dense_matching'
    view_graph_strategy: str = 'from_matching'  # 'from_matching' | 'dense' (clique — all pairs) | 'linear' (sequential chain (k0,k1),(k1,k2),...)
    both_merge_strategy: str = 'concatenate'  # 'concatenate' (boundary-aware filter) | 'separate' (two passes + merge)
    always_add_last_frame: bool = True
    edge_strategy: str = 'on_unreliable'  # 'always' | 'on_unreliable'
    passthrough_skip: int = 1
    min_certainty_threshold: float = 0.975
    certainty_threshold_strategy: str = 'otsu'  # 'otsu' | 'fixed'
    flow_reliability_threshold: float = 0.5
    min_number_of_reliable_matches: int = 0
    matchability_based_reliability: bool = False
    sample_size: int = 10000
    # 'depth' frame filter: max pixel reprojection error for a match to count as a
    # geometric inlier (trust-but-verify layer on top of matchability, no GT poses needed)
    depth_reprojection_threshold_px: float = 5.0
    # 'vggt_covis' frame filter: trained covisibility head on a frozen VGGT trunk
    # (training/vggt_covis/). Reliability = fraction of the source keyframe's object
    # pixels predicted covisible (sigmoid > covis_prob_threshold) in the target frame;
    # runs on ORIGINAL (unmasked) frames, the mask enters only via score aggregation.
    covis_head_weights_path: str | None = None
    covis_prob_threshold: float = 0.5

    # Matching settings
    filter_matcher: str = 'UFM'
    reconstruction_matcher: str = 'UFM'  # 'UFM' | 'RoMa' | 'SIFT' | 'CoTracker' (point tracks through intermediate frames)
    allow_disk_cache: bool = False
    roma: BaseRomaConfig = field(default_factory=BaseRomaConfig)
    ufm: BaseUFMConfig = field(default_factory=BaseUFMConfig)
    sift: BaseSiftConfig = field(default_factory=BaseSiftConfig)
    tracking: BaseTrackingConfig = field(default_factory=BaseTrackingConfig)
    ransac: RANSACConfig = field(default_factory=RANSACConfig)
    sift_filter_min_matches: int = 100
    sift_filter_good_to_add_matches: int = 450

    # Reconstruction settings
    reconstruction_method: str = 'colmap'  # 'colmap' | 'vggt' | 'vggt_omega' | 'mast3r' | 'pi3' | 'map_anything' | 'sam3d'
    mapper: str = 'pycolmap'
    # Bundle-adjustment backend for the pycolmap mapper: 'caspar' (GPU, pycolmap >= 4.1,
    # ~3-4x faster mapping on A100) or 'ceres' (CPU). 'caspar' silently falls back to
    # Ceres when the installed pycolmap has no Caspar support or no GPU is available,
    # so the default is safe on local CPU-only runs and older venvs.
    ba_backend: str = 'caspar'
    init_with_first_two_images: bool = True
    add_track_merging_matches: bool = True
    use_default_colmap_K: bool = True
    similarity_transformation: str = 'kabsch'
    sift_mapping_num_feats: int = 8192
    sift_mapping_min_matches: int = 15
    filter_points_by_segmentation: bool = False
    # Post-reconstruction track-length filter: delete 3D points observed in fewer than
    # this many keyframes (0 disables). Applied after the segmentation filter. Two-view
    # points are the least verified ones and dominate the floating outliers.
    min_track_length: int = 0
    # If True, delete edges from two_view_geometries whose COLMAP RANSAC verification
    # classified them as degenerate (anything but CALIBRATED=2: DEGENERATE, UNCALIBRATED-F
    # fallback, PLANAR/PANORAMIC homographies) before running the mapper. Motivated by the
    # dynamic-onboarding duplicate-model failure (docs/archive/report_dyn_failure_analysis.md):
    # a single PLANAR_OR_PANORAMIC edge with a texture-aliased pure-translation match field
    # glued two disconnected temporal blocks into an offset duplicate of the object.
    filter_degenerate_two_view_edges: bool = False
    # Pruning policy when filter_degenerate_two_view_edges is True:
    # 'all' deletes every non-CALIBRATED edge (cures duplicates, amputates sparse/planar
    # graphs); 'connectivity' lets degenerate edges confirm but never merge — deletes
    # them inside CALIBRATED-connected groups, keeps temporal-neighbour sole bridges
    # (planar chains), refuses cross-gap sole bridges (the duplicate-maker). See
    # prune_degenerate_two_view_edges() and docs/archive/report_dyn_failure_analysis.md.
    filter_degenerate_edges_mode: str = 'all'
    # VidMap-inspired sequential-consistency gate (docs/archive/report_seqconsistency_gate.md):
    # accept a dense reconstruction match on edge (i,j) only if a point track chained
    # through the intermediate frames agrees with it. The tracked prediction is computed
    # by a secondary PointTrackingMatchingProvider (config.onboarding.tracking); a dense
    # match is compared against the displacement of its nearest tracked query (association
    # radius seq_gate_assoc_px) and kept iff the displacement difference is < seq_gate_tau_px.
    # Dense matches with no chain support within the association radius are rejected;
    # edges left with fewer than seq_gate_min_edge_matches matches are dropped entirely
    # (an unsupported bridge). Only meaningful for dense reconstruction matchers.
    seq_consistency_gate: bool = False
    seq_gate_tau_px: float = 5.0
    seq_gate_assoc_px: float = 16.0
    seq_gate_min_edge_matches: int = 15
    # Ablation: if True, keep matches outside the segmentation mask so background points
    # enter the COLMAP reconstruction (matcher + reconstruction both use background).
    # Default False = standard foreground-only reconstruction.
    reconstruction_use_background_points: bool = False
    # Ablation: randomly permute the keyframe (input image) order before reconstruction,
    # so COLMAP's init pair and the neural methods' input sequence order are shuffled.
    # Order-independent for a complete view graph's pair set; only the ordering changes.
    shuffle_keyframes: bool = False
    shuffle_seed: int = 42
    export_view_graph: bool = False

    vggt_depth_conf_threshold: float = 0.1
    # Optional path to a fine-tuned VGGT checkpoint (state dict or {"model": ...}). When
    # set and reconstruction_method='vggt', the adapter overlays it on the base weights.
    vggt_custom_weights_path: str | None = None
    # Crop all frames to ONE fixed square window covering the union of the object masks
    # (+margin) before feeding VGGT. For masked (black-bg) inputs this replaces a
    # mostly-black frame with an object-filling view; the fixed window is a pure
    # principal-point shift + scale, so predicted extrinsics stay comparable to GT.
    vggt_crop_to_object: bool = False
    vggt_crop_margin: float = 1.2
    # The paper's "VGGT + BA" variant: VGGSfM-tracker tracks + one global bundle
    # adjustment seeded from the feed-forward cameras (demo_colmap.py --use_ba).
    vggt_use_ba: bool = False
    # When the matcher config enables crop_matching, restrict it to the RECONSTRUCTION
    # provider and keep the frame-filter provider on full frames — keyframe selection then
    # stays identical to the non-crop pipeline (isolates match-quality effects from
    # view-graph-topology effects).
    crop_matching_reconstruction_only: bool = False
    pi3_conf_threshold: float = 0.1
    # Native VGGSfM baseline (its own tracker + camera init + BA, run as an unmodified
    # subprocess — see adapters/vggsfm_adapter.py). The dedicated venv interpreter is
    # required because VGGSfM pins pycolmap 3.10 (we run pycolmap 4.x in-process).
    vggsfm_python_bin: str | None = None
    vggsfm_fine_tracking: bool = True

    # Decoupled densification (onboarding/densify.py): after alignment succeeds,
    # re-match the registered keyframes densely (UFM matcher config) and
    # triangulate a denser cloud with the camera poses held FIXED, then filter it
    # (short-track / high-reprojection-error gates + multi-view mask carve). The
    # densified model is saved next to the sparse one and logged to the
    # 'Densified' rerun view; the sparse model remains the primary output.
    densify_reconstruction: bool = False
    densify_sample_size: int | None = None      # None -> sample_size
    densify_min_track_len: int = 2       # 2 = keep all triangulated tracks
    densify_max_reproj_error: float = 4.0  # matches the triangulation gate
    densify_carve: bool = True
    # Carve a point once it projects onto background in this many views
    # (1 = strict visual hull, over-carves silhouette boundaries under mask
    # noise; 2 tolerates isolated single-view mask errors).
    densify_carve_min_bg_views: int = 2
    # VGGT-Omega settings — the released checkpoints are HF-gated (manual approval),
    # so the adapter loads from a local file; no runtime download.
    vggt_omega_weights_path: str = os.environ.get('VGGT_OMEGA_WEIGHTS', 'weights/vggt_omega_1b_512.pt')
    # Omega's depth confidence is unbounded and scene-relative: drop the lowest X% of
    # in-mask confidence values (the released demo's convention) instead of a fixed cut.
    vggt_omega_conf_percentile: float = 20.0
    # Visual-hull carving for neural reconstruction methods (VGGT/Mast3r/MapAnything):
    # keep a 3D point only if it projects inside the segmentation mask in EVERY view,
    # not just its source frame. Applied at the pipeline level via
    # carve_reconstruction_by_masks(). COLMAP is exempt (points are multi-view triangulated).
    multiview_mask_filter: bool = True

    # Map Anything settings
    # Supported backends: mapanything | modular_dust3r | vggt | mast3r | must3r |
    # dust3r | moge | pi3 | pi3x | pow3r | pow3r_ba | anycalib | da3
    # See adapters/map_anything_adapter.py:SUPPORTED_BACKENDS for the authoritative list.
    map_anything_backend: str = 'mapanything'
    map_anything_voxel_fraction: float = 0.01

    # SAM3D settings
    sam3d_checkpoint_path: str = os.environ.get('SAM3D_WEIGHTS', 'weights/SAM3D/hf/')
    sam3d_seed: int = 42

    # Separate merge settings (both_merge_strategy='separate')
    merge_matching_black_background: bool = True   # Zero out background using segmentation mask before matching
    merge_use_procrustes: bool = False     # Use matching-based Procrustes alignment
    merge_refine_with_icp: bool = False    # Run ICP refinement after Procrustes
    merge_icp_centroid_prewarp: bool = False  # Centroid prewarp before ICP (off when Procrustes is used)


@dataclass
class CondensationConfig:
    method: str = 'hart'
    descriptor_model: str = 'dinov2'
    descriptor_mask_detections: bool = True
    min_cls_cosine_similarity: float = 0.15
    min_avg_patch_cosine_similarity: float = 0.15
    patch_descriptors_filtering: bool = True
    whiten_dim: int = 0
    csls_k: int = 10
    augment_with_split_detections: bool = True
    augment_with_train_pbr_detections: bool = True
    augmentations_detector: str = 'sam2'
    split: str = 'onboarding_static'


@dataclass
class DetectionConfig:
    templates_source: str = 'cnns'
    condensation_source: str = '1nn-hart'
    descriptor_model: str = 'dinov2'
    descriptor_mask_detections: bool = True
    detector_name: str = 'sam'
    aggregation_function: str = 'max'
    similarity_metric: str = 'cosine'
    confidence_thresh: float = 0.15
    ood_detection_method: str = 'none'
    cosine_similarity_quantile: float = 0.5
    mahalanobis_quantile: float = 0.95
    lowe_ratio_threshold: float = 1.25
    patch_descriptors_filtering: bool = True
    min_avg_patch_cosine_similarity: float = 0.25
    nms_thresh: float = 0.25
    max_num_instances: int = 100


@dataclass
class VisualizationConfig:
    write_to_rerun: bool = True
    jpeg_quality: int = 75
    large_images_write_frequency: int = 1


@dataclass
class PoseEstimationConfig:
    matcher: str = 'UFM'
    sample_size: int = 10000
    min_certainty_threshold: float = 0.975
    flow_reliability_threshold: float = 0.5
    black_background: bool = True
    max_templates_to_match: int = 10


@dataclass
class RendererConfig:
    camera_position: Tuple[float, float, float] = (0, 0, 5.0)
    camera_up: Tuple[float, float, float] = (0, 1, 0)
    obj_center: Tuple[float, float, float] = (0, 0, 0)
    rendered_image_shape: Tuple[int, int] = (500, 500)
    sigmainv: float = 7000
    features: str = 'deep'
    mesh_normalize: bool = False
    texture_size: int = 1000
    gt_mesh_path: Path = None
    optimize_shape: bool = False
    gt_texture_path: Path = None
    tran_init: Tuple[float] = (0., 0., 0.)
    rot_init: Tuple[float] = (0., 0., 0.)


@dataclass
class GloPoseConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    input: InputConfig = field(default_factory=InputConfig)
    onboarding: OnboardingConfig = field(default_factory=OnboardingConfig)
    condensation: CondensationConfig = field(default_factory=CondensationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    bop: BaseBOPConfig = field(default_factory=BaseBOPConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
