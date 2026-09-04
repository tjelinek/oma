from pathlib import Path
from typing import Dict

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ExifTags
from kornia.image import ImageSize
from scipy.ndimage import uniform_filter
from torch.nn import functional as F
from torchvision import transforms


def get_target_shape(image_path: Path, image_downsample: float = 1.0) -> ImageSize:
    if not image_path.is_file():
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    file_ext = image_path.suffix.lower()

    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # Image file
        image = imageio.v3.imread(image_path)
        return ImageSize(width=int(image_downsample * image.shape[1]),
                         height=int(image_downsample * image.shape[0]))

    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:  # Video file
        cap = cv2.VideoCapture(str(image_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {image_path}.")

        # Get the width and height of the video frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return ImageSize(width=int(image_downsample * width),
                         height=int(image_downsample * height))
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def get_video_length_in_frames(video_path: Path) -> int:
    """
    Gets the total number of frames in a video.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        int: Total number of frames in the video.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"The file {video_path} does not exist.")

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}.")

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def get_nth_video_frame(video_path: Path, frame_number: int, mode: str = 'rgb') -> Image:
    """
    Retrieves the nth frame from a video and returns it as a PIL Image in the specified mode.

    Args:
        video_path (Path): Path to the video file.
        frame_number (int): The frame index to retrieve (0-based).
        mode (str): The mode of the output image ('rgb' or 'grayscale').

    Returns:
        Image: The nth frame as a PIL Image.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"The file {video_path} does not exist.")

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}.")

    # Set the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Frame number {frame_number} could not be retrieved.")

    # Process the frame based on the mode
    if mode == 'rgb':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif mode == 'grayscale':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return Image.fromarray(frame)


def get_video_length(video_path: Path) -> int:
    """
    Retrieves the total number of frames in a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        int: The total number of frames in the video.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"The file {video_path} does not exist.")

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}.")

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Could not determine the frame count for {video_path}.")

    return frame_count


def pad_image(image):
    W, H = image.shape[-2:]
    max_size = max(H, W)
    pad_h = max_size - H
    pad_w = max_size - W
    padding = [(pad_h + 1) // 2, pad_h // 2, pad_w // 2, (pad_w + 1) // 2]  # (top, bottom, left, right)

    image = F.pad(image, padding, mode='constant', value=0)

    return image


def resize_and_filter_image(image, new_width, new_height):
    image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    image = uniform_filter(image, size=(3, 3, 1))

    image_tensor = transforms.ToTensor()(image / 255.0)
    image = image_tensor.unsqueeze(0).float()

    return image


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(255, 255, 255)):
    """
    Overlay a colored mask onto an image with adjustable transparency.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C), dtype uint8.
        mask (np.ndarray): Mask of shape (H, W) or (H, W, 1), with values in [0, 1].
        alpha (float): Blending factor for the mask, where 0 = no effect, 1 = full color overlay.
        color (tuple): RGB tuple for overlay color (default is white).

    Returns:
        np.ndarray: Image with mask overlaid, same shape and dtype as input image.
    """
    mask = np.clip(mask.squeeze(), 0, 1)[..., np.newaxis]

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = np.dstack([image] * 3)

    image = image.astype(np.float32)
    overlay_color = np.ones_like(image) * np.array(color, dtype=np.float32)

    overlay_image = (1 - alpha * mask) * image + (alpha * mask) * overlay_color
    return overlay_image.astype(np.uint8)


def overlay_mask_torch(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5, color=(255, 255, 255)):
    """
    Overlay a colored mask onto an image with adjustable transparency.

    Args:
        image (torch.Tensor): Input image of shape (*, C, H, W) or (*, H, W), dtype uint8 or float.
        mask (torch.Tensor): Mask of shape (*, H, W) or (*, 1, H, W), with values in [0, 1].
        alpha (float): Blending factor for the mask, where 0 = no effect, 1 = full color overlay.
        color (tuple): RGB tuple for overlay color (default is white).

    Returns:
        torch.Tensor: Image with mask overlaid, same shape and dtype as input image.
    """
    # Ensure mask is in range [0, 1] and has shape (*, 1, H, W)
    mask = torch.clamp(mask, 0, 1)
    if mask.dim() == image.dim() - 1:  # mask is (*, H, W)
        mask = mask.unsqueeze(-3)  # Add channel dimension -> (*, 1, H, W)
    elif mask.shape[-3] != 1:  # mask has multiple channels, squeeze to 1
        mask = mask.mean(dim=-3, keepdim=True)

    # Handle grayscale images - convert to RGB
    if image.dim() >= 2 and image.shape[-3] == 1:
        image = image.expand(*image.shape[:-3], 3, *image.shape[-2:])
    elif image.dim() == image.dim() and image.shape[-3] not in [1, 3]:
        # If not 1 or 3 channels, assume it's grayscale without channel dim
        image = image.unsqueeze(-3).expand(*image.shape[:-2], 3, *image.shape[-2:])

    # Convert to float for computation
    original_dtype = image.dtype
    image = image.float()

    # Create overlay color tensor with same device and batch dimensions
    color_tensor = torch.tensor(color, dtype=torch.float32, device=image.device)
    overlay_color = color_tensor.view(1, 3, 1, 1).expand_as(image)

    # Apply overlay
    overlay_image = (1 - alpha * mask) * image + (alpha * mask) * overlay_color

    # Convert back to original dtype
    if original_dtype == torch.uint8:
        overlay_image = torch.clamp(overlay_image, 0, 255).to(torch.uint8)
    else:
        overlay_image = overlay_image.to(original_dtype)

    return overlay_image


def get_intrinsics_from_exif(image_path: Path, err_on_default=False) -> torch.tensor:
    image = Image.open(image_path)
    max_size = max(image.size)
    width, height = image.size

    exif = image.getexif()
    focal = None
    sensor_width_mm = None
    focal_35mm = None
    focal_mm = None

    if exif is not None:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, None)
            if tag_name == 'FocalLength':
                focal_mm = float(value[0]) / float(value[1])  # Handle rational values
            elif tag_name == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
            elif tag_name == 'SensorWidth':
                sensor_width_mm = float(value)

        # If FocalLengthIn35mmFilm is available, use it as a fallback
        if focal_35mm and sensor_width_mm:
            focal = focal_35mm / 35.0 * max_size
        elif focal_mm and sensor_width_mm:
            focal = focal_mm * (width / sensor_width_mm)

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length in EXIF data")

        # Fallback to a prior focal length approximation
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    # Compute intrinsics
    fx = focal
    fy = focal  # Assuming square pixels
    cx = width / 2
    cy = height / 2

    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def otsu_threshold(tensor: torch.Tensor) -> float | None:
    """
    Compute Otsu's threshold for a PyTorch tensor.

    Args:
        tensor: 1D or multi-dimensional PyTorch tensor

    Returns:
        threshold: float value of the Otsu threshold
    """
    # Flatten tensor to 1D
    flat_tensor = tensor.flatten()

    if flat_tensor.numel() < 50:
        return None

    # Compute histogram
    min_val = flat_tensor.min().item()
    max_val = flat_tensor.max().item()

    # Use 256 bins like typical image processing implementations
    bins = 256
    hist = torch.histc(flat_tensor, bins=bins, min=min_val, max=max_val)

    # Bin centers
    bin_edges = torch.linspace(min_val, max_val, bins + 1, device=hist.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative sums
    weight1 = torch.cumsum(hist, dim=0)
    weight2 = torch.cumsum(hist.flip(0), dim=0).flip(0)

    # Class means
    mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1
    mean2 = (torch.cumsum((hist * bin_centers).flip(0), dim=0) / weight2.flip(0)).flip(0)

    # Avoid division by zero
    mask = (weight1[:-1] != 0) & (weight2[1:] != 0)

    # Between-class variance
    variance = torch.where(
        mask,
        weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2,
        torch.tensor(0.0, device=tensor.device)
    )

    # Find threshold index with maximum variance
    idx = torch.argmax(variance)
    threshold = bin_centers[idx]

    return threshold.item()


def decode_rle_list(rle_dict: Dict):
    """Decode RLE with counts as list of integers"""
    h, w = rle_dict['size']
    counts = rle_dict['counts']

    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    flag = 0  # 0 for background, 1 for object

    for count in counts:
        mask[idx:idx + count] = flag
        idx += count
        flag = 1 - flag  # toggle between 0 and 1

    return mask.reshape((h, w), order='F')  # Fortran order


def compute_overlap_ratio(masks: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    masks: [N, H, W]
    targets: [M, H, W]
    Gives NxM matrix that gives share of pixels in masks that intersect targets.
    """

    masks_bool = masks > 0
    targets_bool = targets > 0

    masks_flat = masks_bool.flatten(1).float()  # [N, H*W]
    targets_flat = targets_bool.flatten(1).float()  # [M, H*W]

    intersection = masks_flat @ targets_flat.T  # [N, M]
    mask_areas = masks_flat.sum(dim=1, keepdim=True)  # [N, 1]

    overlap_ratio = intersection / mask_areas
    return overlap_ratio


def compute_target_coverage(masks: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    masks_bool = masks > 0
    targets_bool = targets > 0

    masks_flat = masks_bool.flatten(1).float()  # [N, H*W]
    targets_flat = targets_bool.flatten(1).float()  # [N, H*W]

    intersection = (masks_flat * targets_flat).sum(dim=1)  # [N]
    target_areas = targets_flat.sum(dim=1)  # [N]

    coverage = intersection / target_areas
    return coverage
