import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import pydicom
import os
import torch
import pandas as pd
import matplotlib.patches as patches
from monai.bundle import ConfigParser
import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector



def load_ct_series(series_folder):
    slices = []
    for filename in sorted(os.listdir(series_folder)):
        if filename.endswith(".dcm"):
            filepath = os.path.join(series_folder, filename)
            ds = pydicom.dcmread(filepath)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
                slices.append(ds)
    if not slices:
        raise ValueError(f"No suitable CT DICOM files found in {series_folder}")
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image_volume = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in slices])
    return image_volume, slices
def reorder_boxes(boxes):
    reordered = []
    for box in boxes:
        z1, y1, x1, z2, y2, x2 = box
        reordered.append([x1, y1, z1, x2, y2, z2])
    return reordered


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, -1024, 3071)


def motion_blur(volume, kernel_size=15, blur_fraction=0.5):
    # Apply motion blur to a random subset of slices in the 3D volume
    blurred_volume = np.copy(volume)
    num_slices = volume.shape[0]
    num_blur = max(1, int(num_slices * blur_fraction))
    blur_indices = np.random.choice(num_slices, num_blur, replace=False)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    for i in blur_indices:
        blurred_volume[i] = cv2.filter2D(volume[i], -1, kernel)
    return blurred_volume

# --- CNR calculation ---
def boxes_to_mask(boxes, shape):
    """Creates a binary mask for boxes that intersect the given slice index."""
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        x1, y1, z1, x2, y2, z2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, shape[1] - 1), min(y2, shape[0] - 1)
        mask[y1:y2, x1:x2] = 1
    return mask

def calculate_cnr(ct_slice, nodule_mask, background_mask):
    nodule_pixels = ct_slice[nodule_mask > 0]
    background_pixels = ct_slice[background_mask > 0]
    mean_nodule = np.mean(nodule_pixels) if nodule_pixels.size > 0 else 0
    mean_background = np.mean(background_pixels) if background_pixels.size > 0 else 0
    std_background = np.std(background_pixels) if background_pixels.size > 0 else 1
    if std_background == 0:
        return 0
    return np.abs(mean_nodule - mean_background) / std_background

# --- Evaluate metrics ---
def evaluate_image_quality(original_slice, modified_slice, pred_boxes, gt_boxes):

    detected_mask = boxes_to_mask(pred_boxes, original_slice.shape)
    true_mask = boxes_to_mask(gt_boxes, original_slice.shape)
    background_mask = np.logical_not(true_mask)

    psnr_value = psnr(original_slice, modified_slice, data_range=original_slice.max() - original_slice.min())
    cnr_value = calculate_cnr(modified_slice, detected_mask, background_mask)

    return {"PSNR": psnr_value, "CNR": cnr_value}

def draw_boxes(locations, slice_idx, slice, title, box_color='red'):
    _, ax = plt.subplots(1)
    ax.imshow(slice, cmap='gray')  # Keep grayscale rendering

    for box in locations:
        x1, y1, z1, x2, y2, z2 = box
        if z1 <= slice_idx <= z2:
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_dicom_origin_spacing(slices):
    origin = slices[0].ImagePositionPatient  # (x, y, z)
    spacing = list(slices[0].PixelSpacing) + [
        float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])
    ]
    return np.array(origin, dtype=np.float32), np.array(spacing, dtype=np.float32)

def world_to_voxel(world_coord, origin, spacing):
    return [(world_coord[i] - origin[i]) / spacing[i] for i in range(3)]

def extract_gt_boxes_from_series(csv_path, slices):
    df = pd.read_csv(csv_path)
    seriesuid = slices[0].SeriesInstanceUID  # Extract from DICOM metadata
    print(seriesuid)
    origin, spacing = get_dicom_origin_spacing(slices)

    gt_boxes = []
    for _, row in df[df["seriesuid"] == seriesuid].iterrows():
        center_world = [row["coordX"], row["coordY"], row["coordZ"]]
        diameter = row["diameter_mm"]
        center_voxel = world_to_voxel(center_world, origin, spacing)
        half = diameter / 2.0 / spacing  # mm to voxel space
        x1, y1, z1 = [c - h for c, h in zip(center_voxel, half)]
        x2, y2, z2 = [c + h for c, h in zip(center_voxel, half)]
        gt_boxes.append([x1, y1, z1, x2, y2, z2])
    return gt_boxes

def resample_volume(ct_volume, original_spacing, target_spacing=(0.703125, 0.703125, 1.25)):
    from monai.transforms import Spacing
    import torch

    spacing_transform = Spacing(
        pixdim=target_spacing,
        mode='bilinear',
        dtype=np.float32,
    )

    ct_tensor = torch.tensor(ct_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Assumes input volume is in (1, 1, D, H, W)
    resampled = spacing_transform(ct_tensor, original_spacing)
    return resampled.squeeze().numpy()

def run_monai_inference(ct_volume, model_path="c:/Users/Habib/Desktop/CT-lung-nodules-detection/model.pt", device="cpu"):

    # Preprocess volume
    ct_volume = np.clip(ct_volume, -1000, 400)
    ct_volume = (ct_volume + 1000) / 1400  # normalize to 0–1
    ct_tensor = torch.tensor(ct_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (B, C, D, H, W)

    config = ConfigParser()
    config.read_config("c:/Users/Habib/Desktop/CT-lung-nodules-detection/monai_lung_nodule_ct_detection_0.6.8/configs/inference.json")

    # Load detector and inferer
    detector = config.get_parsed_content("detector")
    inferer = config.get_parsed_content("inferer")

    # Load model weights into the underlying network
    detector.network.load_state_dict(torch.load(model_path, map_location=device))
    detector.network.to(device)
    detector.eval()

    # Run inference
    with torch.no_grad():
        output = inferer(inputs=ct_tensor, network=detector.network)[0] 
        
    boxes= output["boxes"].cpu().numpy()
    scores = output["labels_scores"].cpu().numpy()
    return boxes, scores
    
def evaluate_boxes(pred_boxes, gt_boxes, dist_thresh=5.0):
    def center(box):
        x1, y1, _, x2, y2, _ = box
        return np.array([(x1 + x2)/2, (y1 + y2)/2])

    matched = 0
    used_gt = set()
    for pred in pred_boxes:
        pred_center = center(pred)
        for i, gt in enumerate(gt_boxes):
            if i in used_gt:
                continue
            gt_center = center(gt)
            dist = np.linalg.norm(pred_center - gt_center)
            if dist <= dist_thresh:
                matched += 1
                used_gt.add(i)
                break
    sensitivity = matched / len(gt_boxes) if gt_boxes else 0
    return {"Matched": matched, "Total": len(gt_boxes), "Sensitivity": sensitivity}

def convert_boxes_to_voxel_indices(boxes_world, origin, spacing):
    boxes_voxel = []
    for box in boxes_world:
        x1, y1, z1, x2, y2, z2 = box
        start_voxel = world_to_voxel([x1, y1, z1], origin, spacing)
        end_voxel = world_to_voxel([x2, y2, z2], origin, spacing)
        boxes_voxel.append(start_voxel + end_voxel)  # [x1, y1, z1, x2, y2, z2]
    return boxes_voxel
def filter_uncertain_detections(pred_boxes, scores):
    certain_boxes=[]
    for i in range (len(scores)):
        if scores[i] >0.9:
            certain_boxes.append(pred_boxes[i])
    return certain_boxes
    
def add_ring_artifact(slice, num_rings=5, intensity=500, thickness=5):
    """Add strong, thick concentric ring artifacts to a 2D CT slice."""
    output = slice.copy()
    center = (slice.shape[1] // 2, slice.shape[0] // 2)
    max_radius = min(center)
    radii = np.linspace(max_radius // 6, max_radius, num_rings)
    for r in radii:
        rr, cc = np.ogrid[:slice.shape[0], :slice.shape[1]]
        mask = np.abs(np.sqrt((cc - center[0]) ** 2 + (rr - center[1]) ** 2) - r) < thickness
        output[mask] += intensity  # All rings are brighter
    return np.clip(output, -1024, 3071)

def main():
    # Set the path to a single CT series folder (containing DICOM slices)
    series_folder = r"c:/Users/Habib/Desktop/CT-lung-nodules-detection/manifest-1746278435659/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192"
    ct_volume, slices = load_ct_series(series_folder)
    print("Loaded CT volume shape:", ct_volume.shape)
    ct_volume= ct_volume[80:96]
    ct_slice = ct_volume[8]
    slice_idx= 89 #in the original whole volume
    plt.figure(figsize=(6,6))
    plt.imshow(ct_slice, cmap='gray')
    plt.title("Original CT Slice")
    plt.axis('off')
    plt.show()
    
    gt_boxes = extract_gt_boxes_from_series("c:/Users/Habib/Desktop/CT-lung-nodules-detection/annotations.csv", slices)


    print("Found", len(gt_boxes), "ground truth boxes.")
    draw_boxes(gt_boxes, slice_idx, ct_slice, "Ground Truth Lung Nodules")
    
    print("==> Running inference on original volume")

    pred_boxes, scores= run_monai_inference(ct_volume, model_path="c:/Users/Habib/Desktop/CT-lung-nodules-detection/model.pt")
    certain_boxes = filter_uncertain_detections(pred_boxes, scores)
    certain_boxes= reorder_boxes(certain_boxes)
    draw_boxes(certain_boxes, 8,ct_slice, "Detected Lung Nodules", box_color='green')
    eval_result = evaluate_boxes(certain_boxes, gt_boxes)
    print("Evaluation:", eval_result)
    
    image_quality = evaluate_image_quality(ct_slice, ct_slice, certain_boxes, gt_boxes)
    print("Image Quality (Clean):", image_quality)

    print("\n==> Adding noise and evaluating again")
    noisy_ct = add_gaussian_noise(ct_volume, sigma=700)
    noisy_slice= noisy_ct[8]
    plt.imshow(noisy_slice,cmap='gray')
    plt.title("Noisy CT Slice")
    plt.show()
    noisy_pred, scores = run_monai_inference(noisy_ct, model_path="c:/Users/Habib/Desktop/CT-lung-nodules-detection/monai_lung_nodule_ct_detection_0.6.8/models/model.pt")
    certain_boxes= filter_uncertain_detections(noisy_pred, scores)
    certain_boxes= reorder_boxes(certain_boxes)
    
    draw_boxes(certain_boxes, 8, noisy_slice,"Detection After Noise", box_color='orange')
    eval_result_noise = evaluate_boxes(certain_boxes, gt_boxes)
    print("Noisy Evaluation:", eval_result_noise)
    noisy_quality = evaluate_image_quality(ct_slice, noisy_slice, certain_boxes, gt_boxes)
    print("Image Quality (Noisy):", noisy_quality)
    print("\n==> Adding motion artifact and evaluating again")
    # Add motion blur
    blurred_volume = motion_blur(ct_volume)
    blurred_slice= blurred_volume[8]
    plt.imshow(blurred_slice, cmap='gray')
    plt.title("Motion Blurred Slice")
    plt.show()
    blurred_pred, scores = run_monai_inference(blurred_volume, model_path="c:/Users/Habib/Desktop/CT-lung-nodules-detection/monai_lung_nodule_ct_detection_0.6.8/models/model.pt")
    certain_boxes= filter_uncertain_detections(blurred_pred, scores)
    certain_boxes= reorder_boxes(certain_boxes)
    
    draw_boxes(certain_boxes, 8, blurred_slice,"Detection with Motion Artifact", box_color='orange')
    eval_result_blurred = evaluate_boxes(certain_boxes, gt_boxes)
    print("Motion Artifacts Evaluation:", eval_result_blurred)
    noisy_quality = evaluate_image_quality(ct_slice, blurred_slice, certain_boxes, gt_boxes)
    print("Image Quality (Noisy):", noisy_quality)
    
    print("\n==> Adding ring artifact and evaluating again")
    ring_artifact_slice = add_ring_artifact(ct_slice)
    plt.imshow(ring_artifact_slice, cmap='gray')
    plt.title("Ring Artifact Slice")
    plt.show()
    # For demonstration, just run detection on the single slice with ring artifact
    # (You could also apply to the whole volume if desired)
    ring_artifact_volume = ct_volume.copy()
    ring_artifact_volume[8] = ring_artifact_slice
    ring_pred, scores = run_monai_inference(ring_artifact_volume, model_path="c:/Users/Habib/Desktop/CT-lung-nodules-detection/monai_lung_nodule_ct_detection_0.6.8/models/model.pt")
    certain_boxes = filter_uncertain_detections(ring_pred, scores)
    certain_boxes = reorder_boxes(certain_boxes)
    draw_boxes(certain_boxes, 8, ring_artifact_slice, "Detection with Ring Artifact", box_color='purple')
    eval_result_ring = evaluate_boxes(certain_boxes, gt_boxes)
    print("Ring Artifact Evaluation:", eval_result_ring)
    ring_quality = evaluate_image_quality(ct_slice, ring_artifact_slice, certain_boxes, gt_boxes)
    print("Image Quality (Ring Artifact):", ring_quality)

if __name__ == "__main__":
    main()

