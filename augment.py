import torch
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Define the transform_data function
def transform_data(X, Y):
    # settings
    translation = 0.25
    augm_prob = 1.0

    if random.uniform(0.0, 1.0) > augm_prob:
        # no augmentation
        return X, Y

    # ---- 1) roll in H (dim=1) ----
    shift_frac_h = random.uniform(-translation, translation)
    
    shift_hX = int(X.shape[1] * shift_frac_h)
    shift_hY = int(Y.shape[1] * shift_frac_h)
    
    X = torch.roll(X, shifts=shift_hX, dims=1)
    Y = torch.roll(Y, shifts=shift_hY, dims=1)
    if shift_hY > 0:
        X[:, :shift_hX, :] = 0
        Y[:, :shift_hY, :] = 0
    elif shift_hY < 0:
        X[:, shift_hX:, :] = 0
        Y[:, shift_hY:, :] = 0

    # ---- 2) roll in W (dim=2) ----
    shift_frac_w = random.uniform(-translation, translation)
    
    shift_wX = int(X.shape[2] * shift_frac_w)
    shift_wY = int(Y.shape[2] * shift_frac_w)
    
    X = torch.roll(X, shifts=shift_wX, dims=2)
    Y = torch.roll(Y, shifts=shift_wY, dims=2)
    if shift_wY > 0:
        X[:, :, :shift_wX] = 0
        Y[:, :, :shift_wY] = 0
    elif shift_wY < 0:
        X[:, :, shift_wX:] = 0
        Y[:, :, shift_wY:] = 0
    

    # ---- 3) rotation ----
    angle = random.uniform(0, 360)
    X = TF.rotate(X, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    Y = TF.rotate(Y, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)

    # ---- 4) brightness scaling ----
    X = random.uniform(0.5, 1.5) * X

    return X, Y

def main():
    # Create a synthetic grayscale tensor (gradient)
    H, W = 256, 256
    # Horizontal gradient from 0 to 1
    orig_np = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)

    orig_np = torch.ones((H, W))
    orig_np[56:200,56:200] = 0.5
    X = orig_np.unsqueeze(0)  # shape [1, H, W]
    Y = X.clone()

    # Apply augmentation
    X_aug, Y_aug = transform_data(X, Y)

    # Convert to numpy for plotting
    orig_img = X.squeeze(0).cpu().numpy()
    aug_img = X_aug.squeeze(0).cpu().numpy()

    # Plot original vs augmented
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Tensor')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(aug_img, cmap='gray', vmin=0, vmax=1)
    plt.title('Augmented Tensor')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()