import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio as psnr
import pydicom
import os
import torch
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.networks.nets import resnet50


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


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, -1024, 3071)


def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def segment_lungs(ct_slice):
    binary = ct_slice < -400
    binary = ndi.binary_closing(binary, structure=np.ones((5,5)))
    label, num_features = ndi.label(binary)
    sizes = ndi.sum(binary, label, range(num_features + 1))
    mask_size = sizes < (np.max(sizes) * 0.5)
    remove_pixel = mask_size[label]
    binary[remove_pixel] = 0
    return binary


def detect_nodules(ct_slice, lung_mask):
    nodule_candidates = np.logical_and(ct_slice > -600, lung_mask)
    labeled_nodules, num_features = ndi.label(nodule_candidates)
    return labeled_nodules, num_features

# --- CNR calculation ---
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
def evaluate(modified_image, original_image, detected_nodules, true_nodules):
    psnr_value = psnr(original_image, modified_image, data_range=original_image.max() - original_image.min())
    background_mask = np.logical_not(true_nodules)
    cnr_value = calculate_cnr(modified_image, detected_nodules, background_mask)
    detected_voxels = np.sum(detected_nodules > 0)
    true_voxels = np.sum(true_nodules > 0)
    sensitivity = detected_voxels / true_voxels if true_voxels > 0 else 0
    return {"PSNR": psnr_value, "CNR": cnr_value, "Sensitivity": sensitivity}

# --- Simulate artifacts and evaluate ---
def simulate_artifacts_and_evaluate(ct_slice, lung_mask, original_nodules):
    results = {}
    noise_levels = [10, 20, 30]
    for sigma in noise_levels:
        noisy = add_gaussian_noise(ct_slice, sigma)
        noisy_nodules, _ = detect_nodules(noisy, lung_mask)
        results[f'Noise_{sigma}'] = evaluate(noisy, ct_slice, noisy_nodules, original_nodules)
    blurred = motion_blur(ct_slice, kernel_size=15)
    blurred_nodules, _ = detect_nodules(blurred, lung_mask)
    results['MotionBlur'] = evaluate(blurred, ct_slice, blurred_nodules, original_nodules)
    return results

# --- MONAI-based Nodule Segmentation and Detection ---
# NOTE: You must download the MONAI model weights file (monai_lidc_nodule_segmentation.pth) and place it in the project directory for this to work.
def preprocess_volume_monai(volume):
    volume = np.clip(volume, -1000, 400)  # Hounsfield range for lung
    volume = (volume - volume.min()) / (volume.max() - volume.min())  # Normalize to 0–1
    input_tensor = torch.tensor(volume[None, None, :, :, :])  # [B, C, D, H, W]
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=(128, 128, 128), mode='trilinear')
    return input_tensor.float()

def load_monai_model():
    # Build the backbone
    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=(2, 2, 1),
        conv1_t_size=(7, 7, 7)
    )
    # Build the feature extractor
    feature_extractor = resnet_fpn_feature_extractor(
        backbone, 3, False, [1, 2], None
    )
    # Build the RetinaNet model
    model = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=(16, 16, 8),
        use_list_output=False
    )
    # Load the weights
    weights = torch.load("C:/Users/Habib/Desktop/CT-lung-nodules-detection/monai_lung_nodule_ct_detection_0.6.8/models/model.pt", map_location="cpu")
    # Some MONAI checkpoints are dicts with a 'model' key, some are just state_dicts
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.eval()
    return model

def predict_nodules_monai(model, volume_tensor, score_thresh=0.2):
    model.eval()
    with torch.no_grad():
        output = model(volume_tensor)
        # Handle different output types
        if isinstance(output, (list, tuple)):
            if len(output) == 2 and isinstance(output[1], list):
                detection = output[1][0]
            elif isinstance(output[0], dict):
                detection = output[0]
            elif isinstance(output[0], list):
                detection = output[0][0]
            else:
                raise RuntimeError(f"Unexpected output structure: {type(output)}, {output}")
        elif isinstance(output, dict):
            detection = output
        else:
            raise RuntimeError(f"Unexpected output type: {type(output)}")
        print("Detection dict:", detection)
        print("Detection dict keys:", detection.keys())
        # Now try to access boxes and scores
        boxes = detection.get('boxes', None)
        scores = detection.get('scores', None)
        if boxes is None or scores is None:
            print("No 'boxes' or 'scores' in detection dict. Detection dict:", detection)
            return np.array([]), np.array([])
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
    keep = scores >= score_thresh
    return boxes[keep], scores[keep]

def draw_boxes_on_slice(ct_slice, boxes, scores, slice_idx, score_thresh=0.01):
    img = np.stack([ct_slice]*3, axis=-1)  # grayscale to RGB
    for box, score in zip(boxes, scores):
        if score < score_thresh:
            continue
        z1, y1, x1, z2, y2, x2 = box.astype(int)
        if z1 <= slice_idx <= z2:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    return img

# --- Main execution (replace nodule mask with MONAI prediction) ---
def main():
    # Set the path to a single CT series folder (containing DICOM slices)
    series_folder = r"C:\Users\Habib\Desktop\CT-lung-nodules-detection\manifest-1746278435659\LIDC-IDRI\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919"
    ct_volume, dicom_slices = load_ct_series(series_folder)
    print("Loaded CT volume shape:", ct_volume.shape)
    # Pick a middle slice for demonstration
    mid_slice_idx = ct_volume.shape[0] // 2
    ct_slice = ct_volume[mid_slice_idx]
    plt.figure(figsize=(6,6))
    plt.imshow(ct_slice, cmap='gray')
    plt.title("Original CT Slice")
    plt.axis('off')
    plt.show()
    # --- MONAI nodule detection ---
    volume_tensor = preprocess_volume_monai(ct_volume)
    model = load_monai_model()
    boxes, scores = predict_nodules_monai(model, volume_tensor)
    # Visualize bounding boxes on the middle slice
    img_with_boxes = draw_boxes_on_slice(ct_slice, boxes, scores, mid_slice_idx)
    plt.figure(figsize=(6,6))
    plt.imshow(img_with_boxes)
    plt.title("Detected Nodules (Bounding Boxes)")
    plt.axis('off')
    plt.show()
    # Simulate artifacts and evaluate
    noise_levels = [10, 20, 30]
    for sigma in noise_levels:
        noisy = add_gaussian_noise(ct_slice, sigma)
        plt.figure(figsize=(6,6))
        plt.imshow(noisy, cmap='gray')
        plt.title(f"Noisy CT Slice (σ={sigma})")
        plt.axis('off')
        plt.show()
        # You can run detection on noisy slice if you want (single slice, not full volume)
        # Or just compute PSNR
        psnr_value = psnr(ct_slice, noisy, data_range=ct_slice.max() - ct_slice.min())
        print(f"PSNR for σ={sigma}: {psnr_value:.2f} dB")
    blurred = motion_blur(ct_slice, kernel_size=15)
    plt.figure(figsize=(6,6))
    plt.imshow(blurred, cmap='gray')
    plt.title('Motion Blurred CT Slice')
    plt.axis('off')
    plt.show()
    psnr_blur = psnr(ct_slice, blurred, data_range=ct_slice.max() - ct_slice.min())
    print(f"PSNR for motion blur: {psnr_blur:.2f} dB")

    # --- Old nodule detection code (commented for reference) ---
    # lung_mask = segment_lungs(ct_slice)
    # original_nodules = true_nodule_mask[mid_slice_idx]
    # num_nodules = np.max(original_nodules)
    # print(f"Original nodules detected: {num_nodules}")
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1)
    # plt.title('CT Slice')
    # plt.imshow(ct_slice, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1,3,2)
    # plt.title('Lung Mask')
    # plt.imshow(lung_mask, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1,3,3)
    # plt.title('Detected Nodules')
    # plt.imshow(original_nodules, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()

