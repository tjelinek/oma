from typing import Final

axes = ['x', 'y', 'z']


class RerunAnnotations:
    # Input
    template_image: Final[str] = '/observations/template_image'
    template_image_segmentation: Final[str] = '/observations/template_image/segment'
    observed_image: Final[str] = '/observations/observed_image'
    observed_image_segmentation: Final[str] = '/observations/observed_image/segment'
    observed_image_all: Final[str] = '/observations/observed_image_all'
    observed_image_segmentation_all: Final[str] = '/observations/observed_image_all/segment'

    # Detection
    detection_image: Final[str] = '/observations/template_image'
    detection_nearest_neighbors: Final[str] = '/observations/template_image_neighbors'

    # Keyframes
    keyframe_images = '/keyframes/keyframe_images'
    keyframe_graph = '/keyframes/keyframe_graph'

    # Visibility Graph
    view_graph = '/keyframes/view_graph'

    # Matching
    matches_high_certainty: Final[str] = '/matching/high_certainty'
    matches_low_certainty: Final[str] = '/matching/low_certainty'
    matching_certainty: Final[str] = '/matching/certainty'
    matches_high_certainty_segmentation: Final[str] = '/matching/high_certainty/segmentation'
    matches_low_certainty_segmentation: Final[str] = '/matching/low_certainty/segmentation'
    matching_reliability_plot: Final[str] = '/matching/reliability_plot'
    matching_reliability: Final[str] = '/matching/reliability_plot/reliability'
    matching_reliability_threshold_roma: Final[str] = '/matching/reliability_plot/reliability_threshold'

    # Depth frame filter: estimated-vs-GT relative pose error plots
    depth_pose_error_plot: Final[str] = '/matching/depth_pose_error_plot'
    depth_rotation_error: Final[str] = '/matching/depth_pose_error_plot/rotation_error_deg'
    depth_translation_error: Final[str] = '/matching/depth_pose_error_plot/translation_error_deg'

    matchability: Final[str] = 'matching/matchability'
    matches_high_certainty_matchable: Final[str] = '/matching/high_certainty_matchable'
    matches_low_certainty_matchable: Final[str] = '/matching/low_certainty_matchable'
    matching_matchability_plot: Final[str] = '/matching/matchability_plot'
    matching_matchability_plot_share_matchable: Final[str] = '/matching/matchability_plot/share_matchable'
    matching_min_roma_certainty_plot: Final[str] = '/matching/min_roma_certainty_plot/'
    matching_min_roma_certainty_plot_min_certainty: Final[str] = '/matching/min_roma_certainty_plot/min_certainty'

    matches_sift: Final[str] = '/matching/reliability_plot/sift_num_matches/'
    min_matches_sift: Final[str] = '/matching/reliability_plot/min_matches_sift'
    good_to_add_number_of_matches_sift: Final[str] = '/matching/reliability_plot/good_to_add_matches_sift'

    # Observations
    space_visualization: Final[str] = '/3d_space'
    colmap_visualization: Final[str] = '/3d_colmap'
    space_gt_mesh: Final[str] = '/3d_space/gt_mesh'
    space_gt_camera_pose: Final[str] = '/3d_space/gt_camera_pose'
    space_predicted_camera_pose: Final[str] = '/3d_space/predicted_camera_pose'
    space_gt_camera_track: Final[str] = '/3d_space/gt_camera_track'
    space_predicted_camera_track: Final[str] = '/3d_space/predicted_camera_track'
    space_predicted_camera_keypoints: Final[str] = '/3d_space/predicted_camera_keypoints'
    space_predicted_closest_keypoint: Final[str] = '/3d_space/predicted_closest_keypoint'
    space_predicted_reliable_templates: Final[str] = '/3d_space/predicted_reliable_templates'

    # 3D Space — GT model & reconstruction overlay
    space_reconstruction_pointcloud: Final[str] = '/3d_space/reconstruction_pointcloud'

    # Graph comparison
    colmap_covisibility_graph: Final[str] = '/graphs/colmap_covisibility'
    initial_viewgraph: Final[str] = '/graphs/initial_viewgraph'

    # COLMAP
    colmap_pointcloud: Final[str] = '/3d_colmap/pointcloud'
    colmap_pointcloud_densified: Final[str] = '/3d_colmap/pointcloud_densified'
    colmap_gt_camera_pose: Final[str] = '/3d_colmap/gt_camera_pose'
    colmap_gt_camera_track: Final[str] = '/3d_colmap/gt_camera_track'
    colmap_pred_camera_track: Final[str] = '/3d_colmap/pred_camera_track'
    colmap_predicted_camera_poses: Final[str] = '/3d_colmap/predicted_camera_keypoints'
    colmap_predicted_reliable_templates: Final[str] = '/3d_colmap/predicted_reliable_templates'
    colmap_predicted_line_strips_reliable: Final[str] = '/3d_colmap/line_strips_reliable'
    colmap_point_projections: Final[str] = '/3d_colmap/point_projections'

    observed_flow: Final[str] = '/observed_flow/observed_flow'
    observed_flow_occlusion: Final[str] = '/observed_flow/occlusion'
    observed_flow_uncertainty: Final[str] = '/observed_flow/uncertainty'
    observed_flow_with_uncertainty: Final[str] = '/observed_flow/observed_flow_front_uncertainty'
    observed_flow_errors: Final[str] = '/observed_flow/observed_flow_gt_disparity'

    # Optimized model visualizations
    optimized_model_occlusion: Final[str] = '/optimized_values/occlusion'
    optimized_model_render: Final[str] = '/optimized_values/rendering'

    ransac_stats: Final[str] = '/epipolar/ransac_stats'
    ransac_stats_visible: Final[str] = '/epipolar/ransac_stats/visible'
    ransac_stats_predicted_as_visible: Final[str] = '/epipolar/ransac_stats/predicted_as_visible'
    ransac_stats_correctly_predicted_flows: Final[str] = '/epipolar/ransac_stats/correctly_predicted_flows'
    ransac_stats_ransac_predicted_inliers: Final[str] = '/epipolar/ransac_stats/ransac_predicted_inliers'
    ransac_stats_correctly_predicted_inliers: Final[str] = '/epipolar/ransac_stats/correctly_predicted_inliers'
    ransac_stats_ransac_inlier_ratio: Final[str] = '/epipolar/ransac_stats/ransac_inlier_ratio'

    # Pose
    pose_estimation_timing: Final[str] = '/pose/timing/'
    pose_estimation_time: Final[str] = '/pose/timing/pose_estimation_time'
    long_short_chain_remaining_pts: Final[str] = '/pose/chaining/remaining_pts/remaining_percent'
    long_short_chain_diff_template: Final[str] = '/pose/chaining/template'
    chained_pose_polar: Final[str] = '/pose/chaining/polar_angle'
    chained_pose_polar_template: Final[str] = '/pose/chaining/polar_angle/template'
    chained_pose_long_flow_polar: Final[str] = '/pose/chaining/polar_angle/long_flow'
    chained_pose_short_flow_polar: Final[str] = '/pose/chaining/polar_angle/short_flow'
    long_short_chain_diff: Final[str] = '/pose/chaining/polar_angle/difference'

    chained_pose_long_flow: Final[str] = '/pose/chaining/long_flow'
    chained_pose_long_flow_template: Final[str] = '/pose/chaining/long_flow/template'
    chained_pose_long_flow_axes: Final[dict] = {
        axis: f'/pose/chaining/long_flow/{axis}_axis' for axis in axes
    }

    chained_pose_short_flow: Final[str] = '/pose/chaining/short_flow'
    chained_pose_short_flow_template: Final[str] = '/pose/chaining/short_flow/template'
    chained_pose_short_flow_axes: Final[dict] = {
        axis: f'/pose/chaining/short_flow/{axis}_axis' for axis in axes
    }

    cam_delta_r_short_flow: Final[str] = '/pose/cam/rot/short_flow/'
    cam_delta_r_short_flow_template = '/pose/cam/rot/short_flow/template'
    cam_delta_r_short_flow_zaragoza: Final[str] = '/pose/cam/rot/short_flow/cam_pose_delta_zaragoza'
    cam_delta_r_short_flow_RANSAC: Final[str] = '/pose/cam/rot/short_flow/cam_pose_delta_RANSAC'

    cam_delta_t_short_flow: Final[str] = '/pose/cam/tran/short_flow/'
    cam_delta_t_short_flow_template = '/pose/cam/tran/short_flow/template'
    cam_delta_t_short_flow_zaragoza: Final[str] = '/pose/cam/tran/short_flow/cam_pose_delta_zaragoza'
    cam_delta_t_short_flow_RANSAC: Final[str] = '/pose/cam/tran/short_flow/cam_pose_delta_RANSAC'

    cam_delta_r_long_flow: Final[str] = '/pose/cam/rot/long_flow/'
    cam_delta_r_long_flow_template = '/pose/cam/rot/long_flow/template'
    cam_delta_r_long_flow_zaragoza: Final[str] = '/pose/cam/rot/long_flow/cam_pose_delta_zaragoza'
    cam_delta_r_long_flow_RANSAC: Final[str] = '/pose/cam/rot/long_flow/cam_pose_delta_RANSAC'

    cam_delta_t_long_flow: Final[str] = '/pose/cam/tran/long_flow/'
    cam_delta_t_long_flow_template = '/pose/cam/tran/long_flow/template'
    cam_delta_t_long_flow_zaragoza: Final[str] = '/pose/cam/tran/long_flow/cam_pose_delta_zaragoza'
    cam_delta_t_long_flow_RANSAC: Final[str] = '/pose/cam/tran/long_flow/cam_pose_delta_RANSAC'

    obj_rot_1st_to_last: Final[str] = '/pose/rotation'
    obj_rot_1st_to_last_axes: Final[dict] = {
        axis: f'/pose/rotation/{axis}_axis' for axis in axes
    }
    obj_rot_1st_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/rotation/{axis}_axis_gt' for axis in axes
    }

    obj_tran_1st_to_last: Final[str] = '/pose/translation'
    obj_tran_1st_to_last_axes: Final[dict] = {
        axis: f'/pose/translation/{axis}_axis' for axis in axes
    }
    obj_tran_1st_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/translation/{axis}_axis_gt' for axis in axes
    }

    cam_rot_ref_to_last: Final[str] = '/pose/cam_rot_ref_to_last'
    cam_rot_ref_to_last_template: Final[str] = '/pose/cam_rot_ref_to_last/template'
    cam_rot_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/cam_rot_ref_to_last/{axis}_axis' for axis in axes
    }
    cam_rot_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/cam_rot_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    cam_tran_ref_to_last: Final[str] = '/pose/cam_tran_ref_to_last'
    cam_tran_ref_to_last_template: Final[str] = '/pose/cam_tran_ref_to_last/template'
    cam_tran_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/cam_tran_ref_to_last/{axis}_axis' for axis in axes
    }
    cam_tran_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/cam_tran_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    obj_rot_ref_to_last: Final[str] = '/pose/obj_rot_ref_to_last'
    obj_rot_ref_to_last_template: Final[str] = '/pose/obj_rot_ref_to_last/template'
    obj_rot_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/obj_rot_ref_to_last/{axis}_axis' for axis in axes
    }
    obj_rot_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/obj_rot_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    obj_tran_ref_to_last: Final[str] = '/pose/obj_tran_ref_to_last'
    obj_tran_ref_to_last_template: Final[str] = '/pose/obj_tran_ref_to_last/template'
    obj_tran_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/obj_tran_ref_to_last/{axis}_axis' for axis in axes
    }
    obj_tran_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/obj_tran_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    translation_scale: Final[str] = '/pose/translation_scale'
    translation_scale_gt_axes: Final[dict] = {
        axis: f'/pose/translation_scale/{axis}_gt' for axis in axes
    }
    translation_scale_estimated: Final[str] = '/pose/translation_scale/estimated'

    # Model Merging
    merge_before: Final[str] = '/merge/before'
    merge_before_target_pointcloud: Final[str] = '/merge/before/target_pointcloud'
    merge_before_target_cameras: Final[str] = '/merge/before/target_cameras'
    merge_before_source_pointcloud: Final[str] = '/merge/before/source_pointcloud'
    merge_before_source_cameras: Final[str] = '/merge/before/source_cameras'
    merge_before_gt_mesh: Final[str] = '/merge/before/gt_mesh'
    merge_before_gt_camera_track: Final[str] = '/merge/before/gt_camera_track'
    merge_after_procrustes: Final[str] = '/merge/after_procrustes'
    merge_after_procrustes_target_pointcloud: Final[str] = '/merge/after_procrustes/target_pointcloud'
    merge_after_procrustes_target_cameras: Final[str] = '/merge/after_procrustes/target_cameras'
    merge_after_procrustes_source_pointcloud: Final[str] = '/merge/after_procrustes/source_pointcloud'
    merge_after_procrustes_source_cameras: Final[str] = '/merge/after_procrustes/source_cameras'
    merge_after_procrustes_gt_mesh: Final[str] = '/merge/after_procrustes/gt_mesh'
    merge_after_procrustes_gt_camera_track: Final[str] = '/merge/after_procrustes/gt_camera_track'
    merge_before_icp: Final[str] = '/merge/before_icp'
    merge_before_icp_target_pointcloud: Final[str] = '/merge/before_icp/target_pointcloud'
    merge_before_icp_target_cameras: Final[str] = '/merge/before_icp/target_cameras'
    merge_before_icp_source_pointcloud: Final[str] = '/merge/before_icp/source_pointcloud'
    merge_before_icp_source_cameras: Final[str] = '/merge/before_icp/source_cameras'
    merge_before_icp_gt_mesh: Final[str] = '/merge/before_icp/gt_mesh'
    merge_before_icp_gt_camera_track: Final[str] = '/merge/before_icp/gt_camera_track'
    merge_after_icp: Final[str] = '/merge/after_icp'
    merge_after_icp_target_pointcloud: Final[str] = '/merge/after_icp/target_pointcloud'
    merge_after_icp_target_cameras: Final[str] = '/merge/after_icp/target_cameras'
    merge_after_icp_source_pointcloud: Final[str] = '/merge/after_icp/source_pointcloud'
    merge_after_icp_source_cameras: Final[str] = '/merge/after_icp/source_cameras'
    merge_after_icp_gt_mesh: Final[str] = '/merge/after_icp/gt_mesh'
    merge_after_icp_gt_camera_track: Final[str] = '/merge/after_icp/gt_camera_track'
    merge_after: Final[str] = '/merge/after'
    merge_after_pointcloud: Final[str] = '/merge/after/pointcloud'
    merge_after_cameras: Final[str] = '/merge/after/cameras'
    merge_after_gt_mesh: Final[str] = '/merge/after/gt_mesh'
    merge_after_gt_camera_track: Final[str] = '/merge/after/gt_camera_track'
    merge_match_pairs: Final[str] = '/merge/match_pairs'
    merge_match_info: Final[str] = '/merge/match_info'
