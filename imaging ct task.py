import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio as psnr
import pydicom
import os

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

# --- Add Gaussian noise ---
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, -1024, 3071)

# --- Simulate motion blur ---
def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

# --- Simple lung segmentation ---
def segment_lungs(ct_slice):
    binary = ct_slice < -400
    binary = ndi.binary_closing(binary, structure=np.ones((5,5)))
    label, num_features = ndi.label(binary)
    sizes = ndi.sum(binary, label, range(num_features + 1))
    mask_size = sizes < (np.max(sizes) * 0.5)
    remove_pixel = mask_size[label]
    binary[remove_pixel] = 0
    return binary

# --- Simple nodule detection ---
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

# --- Main execution ---
def main():
    # Set the path to a single CT series folder (containing DICOM slices)
    series_folder = r"C:\Users\Habib\Desktop\CT-lung-nodules-detection\manifest-1746278435659\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192"
    ct_volume, dicom_slices = load_ct_series(series_folder)
    print("Loaded CT volume shape:", ct_volume.shape)
    # Pick a middle slice for demonstration
    mid_slice_idx = ct_volume.shape[0] // 2
    ct_slice = ct_volume[mid_slice_idx]
    # Visualize the CT slice
    plt.figure(figsize=(6,6))
    plt.imshow(ct_slice, cmap='gray')
    plt.title("Original CT Slice")
    plt.axis('off')
    plt.show()
    # Segment lungs
    lung_mask = segment_lungs(ct_slice)
    # Detect nodules (use as pseudo-ground-truth for demo)
    original_nodules, num_nodules = detect_nodules(ct_slice, lung_mask)
    print(f"Original nodules detected: {num_nodules}")
    # Visualize lung mask and detected nodules
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title('CT Slice')
    plt.imshow(ct_slice, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Lung Mask')
    plt.imshow(lung_mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Detected Nodules')
    plt.imshow(original_nodules, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Simulate artifacts and evaluate
    results = simulate_artifacts_and_evaluate(ct_slice, lung_mask, original_nodules)
    # Print Results
    for condition, metrics in results.items():
        print(f"\nCondition: {condition}")
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        print(f"CNR: {metrics['CNR']:.2f}")
        print(f"Sensitivity: {metrics['Sensitivity']:.2f}")
    # Optional: Visualize one noisy and one blurred example
    noisy = add_gaussian_noise(ct_slice, 20)
    blurred = motion_blur(ct_slice, 15)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(ct_slice, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Noisy (σ=20)')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Motion Blurred')
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

