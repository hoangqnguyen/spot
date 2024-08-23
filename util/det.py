import torch
import matplotlib.pyplot as plt
import numpy as np

def convert_target_to_prediction_shape(target, P):
    B, T, _ = target.shape
    
    # Initialize the output tensor with zeros
    output = torch.zeros((B, T, P, P, 3))

    # Compute the size of each patch in normalized coordinates
    patch_size = 1.0 / P

    # Compute patch indices (vectorized)
    x_idx = torch.clamp((target[..., 0] * P).long(), max=P-1)
    y_idx = torch.clamp((target[..., 1] * P).long(), max=P-1)

    # Compute patch centers (vectorized)
    patch_centers_x = (x_idx.float() + 0.5) * patch_size
    patch_centers_y = (y_idx.float() + 0.5) * patch_size

    # Compute offsets from patch centers (vectorized)
    x_offset = target[..., 0] - patch_centers_x
    y_offset = target[..., 1] - patch_centers_y

    # Print debug information
    # print(f"x_idx: {x_idx}, y_idx: {y_idx}")
    # print(f"patch_centers_x: {patch_centers_x}, patch_centers_y: {patch_centers_y}")
    # print(f"x_offset: {x_offset}, y_offset: {y_offset}")

    # Set object presence flag and offsets in the output tensor
    object_presence = ((target[..., 0] != 0) | (target[..., 1] != 0)).float()
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 0] = object_presence
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 1] = x_offset
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 2] = y_offset

    return output

def visualize_prediction_grid(target, output, P):
    """
    Visualizes the original target positions and the converted offsets on a grid of patches.
    
    Parameters:
    - target: Tensor of shape (B, T, 2) where the last dimension represents (x, y) coordinates in range [0, 1].
    - output: Tensor of shape (B, T, P, P, 3) where the last dimension represents (object_presence, x_offset, y_offset).
    - P: Number of patches per dimension.
    """
    B, T, _ = target.shape
    patch_size = 1.0 / P

    for b in range(B):
        fig, axes = plt.subplots(1, T, figsize=(15, 5))
        
        for t in range(T):
            ax = axes[t]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            # Draw the patch grid
            for i in range(P + 1):
                ax.axhline(i * patch_size, color='gray', linestyle='--', linewidth=0.5)
                ax.axvline(i * patch_size, color='gray', linestyle='--', linewidth=0.5)

            # Plot the original target position
            x, y = target[b, t]
            ax.plot(x, y, 'ro', label='Original Position')

            # Skip if there is no object
            if x == 0 and y == 0:
                continue
            
            # Calculate patch indices
            x_idx = min(int(x * P), P - 1)
            y_idx = min(int(y * P), P - 1)

            # Plot the patch center
            patch_center_x = (x_idx + 0.5) * patch_size
            patch_center_y = (y_idx + 0.5) * patch_size
            ax.plot(patch_center_x, patch_center_y, 'go', label='Patch Center')

            # Plot the offset position within the patch
            x_offset = output[b, t, x_idx, y_idx, 1]
            y_offset = output[b, t, x_idx, y_idx, 2]
            pred_x = patch_center_x + x_offset
            pred_y = patch_center_y + y_offset
            ax.plot(pred_x, pred_y, 'bx', label='Predicted Offset Position')

            ax.legend()
            ax.set_title(f'Frame {t + 1}')

        plt.suptitle(f'Batch {b + 1}')
        plt.show()

def convert_prediction_to_target_shape(prediction, P):
    B, T, _, _, _ = prediction.shape
    
    # Initialize the output tensor
    target = torch.zeros((B, T, 2))
    
    # Compute the size of each patch in normalized coordinates
    patch_size = 1.0 / P

    # Find the patch with the object presence flag set
    object_presence = prediction[..., 0]  # Shape: (B, T, P, P)
    max_indices = object_presence.view(B, T, -1).argmax(-1)  # Shape: (B, T)
    
    # Correctly compute the x and y indices
    x_idx = max_indices // P
    y_idx = max_indices % P
    
    # Get the x and y offsets
    x_offset = prediction[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 1]
    y_offset = prediction[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 2]
    
    # Compute the patch centers (in normalized coordinates)
    patch_centers_x = (x_idx.float() + 0.5) * patch_size
    patch_centers_y = (y_idx.float() + 0.5) * patch_size
    
    # Compute the final (x, y) coordinates
    target[..., 0] = patch_centers_x + x_offset
    target[..., 1] = patch_centers_y + y_offset
    
    return target



if __name__ == '__main__':

    # TEST if the function works correctly

    P = 7

    xys = [[0.6594, 0.4653],[0.1625, 0.6083],[0.4883, 0.4778],[0.3820, 0.4250],[0.5078, 0.4528],[0.7359, 0.5319],[0.7125, 0.4389],[0.6187, 0.3264],[0.5148, 0.3792],[0.5375, 0.5250],[0.5703, 0.4653],[0.5914, 0.4806],[0.5148, 0.5028],[0.6938, 0.5347],[0.6336, 0.5931]]

    for xy in xys:
        target = torch.tensor(xy)[None, None, ...]
        output = convert_target_to_prediction_shape(target, P)
        # visualize_prediction_grid(target, output, P)

        target_reconstructed = convert_prediction_to_target_shape(output, P)

        diff = (target - target_reconstructed).abs().sum()
        print("DIFF=", diff)
        print("="* 20)
        # print(f"target_reconstructed: {target_reconstructed}")