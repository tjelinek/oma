import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from utils.general import get_not_occluded_foreground_points, tensor_index_to_coordinates_xy
from visualizations import flow_viz


def compare_flows_with_images(image1, image2, flow_up, flow_up_prime,
                              gt_silhouette_current=None, gt_silhouette_prev=None):
    """
    Visualizes flow_up and flow_up_prime along with image1 and image2.

    :param image1: uint8 image with 0-255 as color in [C, H, W] format
    :param image2: uint8 image with 0-255 as color in [C, H, W] format
    :param flow_up: np.ndarray [H, W, 2] flow
    :param flow_up_prime: np.ndarray [H, W, 2] flow
    :param gt_silhouette_current: Silhouette of the ground truth in the current frame
    :param gt_silhouette_prev: Silhouette of the ground truth in the previous frame
    :return: PIL Image
    """
    return visualize_flow_with_images(image1, image2, flow_up, flow_up_prime, gt_silhouette_current, gt_silhouette_prev)


def tensor_to_pil_with_alpha(tensor, alpha=0.2):
    # Convert tensor to numpy array and scale to 255 if necessary
    array = (tensor.numpy() * 255).astype(np.uint8)

    # Create an RGBA image with the given alpha value for the segmentation
    rgba_array = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    rgba_array[..., :3] = array[:, :, None]  # Assign the tensor values to red and green channels
    rgba_array[..., 3] = (array > 0) * alpha * 256.0  # Create alpha channel based on tensor values

    # Convert to PIL image
    img = Image.fromarray(rgba_array, 'RGBA')

    return img


def visualize_flow_with_images(templates, image2, observed_flows=None, gt_flows=None, gt_silhouette_current=None,
                               gt_silhouettes_prev=None, flow_occlusion_masks=None):
    """
    Visualize flow with multiple template images.

    :param templates: List of uint8 images with [0-1] color range in [C, H, W] format.
    :param image2: uint8 image with [0-1] color range in [C, H, W] format.
    :param observed_flows: List of np.ndarray [H, W, 2] flow for each template.
    :param gt_flows: np.ndarray [H, W, 2] flow.
    :param gt_silhouette_current: Tensor - silhouette of the ground truth in the current frame of shape [H, W]
    :param gt_silhouettes_prev: List of Tensors  - silhouette of the ground truth in the previous frame of shape [H, W]
    :param flow_occlusion_masks: List of Tensors - indicating flow occlusion in format [H, W].
    """
    width, height = image2.shape[-1], image2.shape[-2]

    assert observed_flows is not None or gt_flows is not None
    assert np.all([(height, width) == t.shape[-2:] for t in templates])
    assert gt_silhouettes_prev is None or len(templates) == len(gt_silhouettes_prev)
    assert flow_occlusion_masks is None or len(templates) == len(flow_occlusion_masks)
    assert observed_flows is None or len(templates) == len(observed_flows)

    transform = T.ToPILImage()
    image2_pil = transform(image2)

    if gt_silhouette_current is not None:
        image2_pil = overlay_silhouette_on_image(gt_silhouette_current, image2_pil)
    draw2 = ImageDraw.Draw(image2_pil)

    canvas_width_coef = int(observed_flows is not None) + int(gt_flows is not None) + 2
    canvas = Image.new('RGBA', (width * canvas_width_coef, len(templates) * height), (255, 255, 255, 255))

    template_pils = []
    for template_idx, template in enumerate(templates):
        # Tensors to PIL Images
        template_pil = transform(template)

        if flow_occlusion_masks is not None:
            flow_occlusion_mask = flow_occlusion_masks[template_idx]
            occlusion_blend = int(255 * 0.25)
            occlusion_mask_pil = tensor_to_pil_with_alpha(flow_occlusion_mask, alpha=1.).convert("L")
            silver_image = Image.new('RGBA', template_pil.size, (255, 255, 255, occlusion_blend))
            template_pil.paste(silver_image, (0, 0), mask=occlusion_mask_pil)

        if gt_silhouettes_prev is not None:
            gt_silhouette_prev = gt_silhouettes_prev[template_idx]
            template_pil = overlay_silhouette_on_image(gt_silhouette_prev, template_pil)

        observed_flow_pil = None
        if observed_flows is not None:
            observed_flow = observed_flows[template_idx]
            if gt_flows is not None:
                observed_flow_image = flow_viz.flow_to_image(np.concatenate([observed_flow, gt_flows], axis=1))
            else:
                observed_flow_image = flow_viz.flow_to_image(observed_flow)
            observed_flow_pil = transform(observed_flow_image)

        gt_flow_pil = None
        if gt_flows is not None:
            gt_flow_image = flow_viz.flow_to_image(gt_flows)
            gt_flow_pil = transform(gt_flow_image)

        draw1 = ImageDraw.Draw(template_pil)

        step = max(width, height) // 20

        r = max(height // 400, 1)  # radius of the drawn point

        for y in range(step, height, step):
            for x in range(step, width, step):
                draw1.ellipse((x - r, y - r, x + r, y + r), fill='black')
                draw2.ellipse((x - r, y - r, x + r, y + r), fill='black')

                if observed_flows is not None:
                    observed_flow = observed_flows[template_idx]

                    shift_up_x = observed_flow[y, x, 0]
                    shift_up_y = observed_flow[y, x, 1]

                    draw2.line((x, y, x + shift_up_x, y + shift_up_y), fill='red')

                    draw2.line((x + shift_up_x, y + shift_up_y - r, x + shift_up_x, y + shift_up_y + r),
                               fill='red')
                    draw2.line((x + shift_up_x - r, y + shift_up_y, x + shift_up_x + r, y + shift_up_y),
                               fill='red')
                if gt_flows is not None:
                    shift_up_prime_x = gt_flows[y, x, 0]
                    shift_up_prime_y = gt_flows[y, x, 1]

                    draw2.line((x, y, x + shift_up_prime_x, y + shift_up_prime_y), fill='green')

                    draw2.line((x + shift_up_prime_x - r, y + shift_up_prime_y - r, x + shift_up_prime_x + r,
                                y + shift_up_prime_y + r), fill='green')
                    draw2.line((x + shift_up_prime_x - r, y + shift_up_prime_y + r, x + shift_up_prime_x + r,
                                y + shift_up_prime_y - r), fill='green')

        canvas.paste(template_pil, (0, template_idx * height))
        if observed_flow_pil is not None:
            canvas.paste(observed_flow_pil, (width, template_idx * height))
        elif gt_flow_pil is not None and observed_flow_pil is None:
            canvas.paste(gt_flow_pil, (width, template_idx * height))
        canvas.paste(image2_pil, ((canvas_width_coef - 1) * width, template_idx * height))

    return canvas


def overlay_silhouette_on_image(silhouette, image_pil):
    silh1_PIL = tensor_to_pil_with_alpha(silhouette, alpha=0.25)
    background1_PIL = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    background1_PIL.paste(silh1_PIL, (0, 0))
    background1_PIL.paste(image_pil, (0, 0))
    image_pil = background1_PIL
    return image_pil


def normalize_rendered_flows(rendered_flows, rendering_width, rendering_height, original_width,
                             original_height):
    rendered_flows[..., 0] = rendered_flows[..., 0] * (rendering_width / original_width)
    rendered_flows[..., 1] = rendered_flows[..., 1] * (rendering_height / original_height)

    return rendered_flows


def flow_image_coords_to_unit_coords(observed_flow: torch.Tensor) -> torch.Tensor:
    # Normalize x-components by width and y-components by height
    observed_flow = observed_flow.clone()
    width = observed_flow.shape[-1]
    height = observed_flow.shape[-2]
    observed_flow[:, :, 0, ...] /= width
    observed_flow[:, :, 1, ...] /= height

    return observed_flow


def flow_unit_coords_to_image_coords(observed_flow: torch.Tensor) -> torch.Tensor:
    # Convert unit range coordinates back to image coordinates
    observed_flow = observed_flow.clone()
    width = observed_flow.shape[-1]
    height = observed_flow.shape[-2]
    observed_flow[:, :, 0, ...] *= width
    observed_flow[:, :, 1, ...] *= height

    return observed_flow


def source_coords_to_target_coords(source_coords: torch.Tensor, flow: torch.Tensor):
    y1, x1 = source_coords.permute(1, 0)
    delta_x, delta_y = flow[0, 0, :, -y1.to(torch.int), x1.to(torch.int)]

    # Compute target coordinates
    x2_f, y2_f = x1 + delta_x, y1 - delta_y

    return torch.stack([y2_f, x2_f], dim=0).permute(1, 0)


def source_to_target_coords_world_coord_system(src_pts_yx: torch.Tensor, flow: torch.Tensor):
    y1, x1 = src_pts_yx.permute(1, 0)
    delta_x, delta_y = flow[0, 0, :, y1.to(torch.int), -x1.to(torch.int)]

    # Compute target coordinates
    y2_f, x2_f = y1 + delta_y, x1 - delta_x

    return torch.stack([y2_f, x2_f], dim=0).permute(1, 0)


def source_coords_to_target_coords_image(source_coords: np.ndarray, flow: np.ndarray):
    y1, x1 = source_coords.T
    delta_x, delta_y = flow[0, 0, :, -y1.astype(int), x1.astype(int)].T

    # Compute target coordinates
    x2_f, y2_f = x1 + delta_x, y1 - delta_y

    return np.stack([y2_f, x2_f], axis=0)


def source_coords_to_target_coords_np(source_coords: np.ndarray, flow: np.ndarray):
    y1, x1 = source_coords.T
    delta_x, delta_y = flow[0, 0, :, y1.astype(int), x1.astype(int)].T

    # Compute target coordinates
    x2_f, y2_f = x1 + delta_x, y1 + delta_y

    return np.stack([y2_f, x2_f], axis=0).T


def get_correct_correspondences_mask(gt_flow, src_pts_yx, dst_pts_yx, epe_threshold):
    dst_pts_yx_gt_flow = source_coords_to_target_coords(src_pts_yx, gt_flow)
    dst_pts_epe = torch.linalg.norm(dst_pts_yx - dst_pts_yx_gt_flow, dim=1)
    ok_pts_indices = torch.nonzero(dst_pts_epe < float(epe_threshold)).squeeze(-1)
    return ok_pts_indices


def get_correct_correspondence_mask_world_system(gt_flow, src_pts_yx, dst_pts_yx, epe_threshold):
    dst_pts_yx_gt_flow = source_to_target_coords_world_coord_system(src_pts_yx, gt_flow)
    dst_pts_epe = torch.linalg.norm(dst_pts_yx - dst_pts_yx_gt_flow, dim=1)
    ok_pts_indices = torch.nonzero(dst_pts_epe < float(epe_threshold)).squeeze(-1)
    return ok_pts_indices


def get_non_occluded_foreground_correspondences(observed_flow_occlusion, observed_flow_segmentation, observed_flow,
                                                segmentation_mask_threshold: float, occlusion_coef_threshold: float):
    src_pts_yx, _ = get_not_occluded_foreground_points(observed_flow_occlusion, observed_flow_segmentation,
                                                       segmentation_mask_threshold, occlusion_coef_threshold)

    dst_pts_yx = source_coords_to_target_coords(src_pts_yx, observed_flow)

    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

    return src_pts_xy, dst_pts_xy


def tensor_image_to_mft_format(image_tensor):
    return image_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def roma_warp_to_pixel_coordinates(coords, H_A, W_A, H_B=None, W_B=None):
    if coords.shape[-1] == 2:
        return _roma_warp_to_pixel_coordinates(coords, H_A, W_A)

    if isinstance(coords, (list, tuple)):
        kpts_A, kpts_B = coords[0], coords[1]
    else:
        kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
    return _roma_warp_to_pixel_coordinates(kpts_A, H_A, W_A), _roma_warp_to_pixel_coordinates(kpts_B, H_B, W_B)


def _roma_warp_to_pixel_coordinates(coords, H, W):
    kpts = torch.stack((W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), dim=-1)
    return kpts


def convert_to_roma_warp(dst_pts_xy_forward: torch.Tensor, dst_pts_xy_backward: torch.Tensor = None) -> torch.Tensor:
    _, H, W = dst_pts_xy_forward.shape

    # Create source pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=dst_pts_xy_forward.device),
        torch.arange(W, dtype=torch.float32, device=dst_pts_xy_forward.device),
        indexing='ij'
    )
    src_pts = torch.stack([x, y], dim=0)  # (2, H, W)

    # Normalize function
    def normalize(pts):
        x = 2.0 * pts[0] / (W - 1) - 1
        y = 2.0 * pts[1] / (H - 1) - 1
        return torch.stack([x, y], dim=0)  # (2, H, W)

    src_norm = normalize(src_pts)
    dst_norm_forward = normalize(dst_pts_xy_forward)

    if dst_pts_xy_backward is not None:
        dst_norm_backward = normalize(dst_pts_xy_forward)

    # Forward: [x_src, y_src, x_dst, y_dst]
    forward = torch.stack([
        src_norm[0], src_norm[1],
        dst_norm_forward[0], dst_norm_forward[1]
    ], dim=-1)  # (H, W, 4)

    # Backward (approximate): [x_dst, y_dst, x_src, y_src]
    if dst_pts_xy_backward is not None:
        backward = torch.stack([
            src_norm[0], src_norm[1],
            dst_norm_backward[0], dst_norm_backward[1]
        ], dim=-1)  # (H, W, 4)
    else:
        backward = torch.stack([
            dst_norm_forward[0], dst_norm_forward[1],
            src_norm[0], src_norm[1]
        ], dim=-1)  # (H, W, 4)

    # Concatenate along width → (H, 2W, 4)
    warp = torch.cat([forward, backward], dim=1)
    return warp


def convert_certainty_to_roma_format(certainty_forward: torch.Tensor, certainty_backward: torch.Tensor = None) -> \
        torch.Tensor:
    H, W = certainty_forward.shape
    roma_certainty = torch.zeros(H, 2 * W, dtype=certainty_forward.dtype, device=certainty_forward.device)
    roma_certainty[:, :W] = certainty_forward
    if certainty_backward is not None:
        roma_certainty[:, W:] = certainty_backward
    else:
        roma_certainty[:, W:] = torch.zeros_like(certainty_forward)

    return roma_certainty
