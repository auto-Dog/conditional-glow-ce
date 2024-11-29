# 将缩略图的颜色变化应用到大图上
import numpy as np
from skimage.transform import resize

def apply_color_transfer(input_thumbnail, output_thumbnail, input_image):
    """
    Apply the color transfer from the thumbnail pair to the original image. All in [0-1] range
    
    Args:
        input_thumbnail (ndarray): Input thumbnail image (e.g., resized original image).
        output_thumbnail (ndarray): Output thumbnail image with altered colors.
        input_image (ndarray): Original input image.

    Returns:
        ndarray: Color-transferred image.
    """
    # # Ensure all inputs are normalized to [0, 1] for calculations
    # if np.max(input_thumbnail)>1.:
    #     input_thumbnail = input_thumbnail / 255.0
    # if np.max(output_thumbnail)>1.:
    #     output_thumbnail = output_thumbnail / 255.0
    # if np.max(input_image)>1.:
    #     input_image = input_image / 255.0

    # Resize thumbnails to match original image dimensions for pixel-wise processing
    resized_input_thumbnail = resize(input_thumbnail, input_image.shape[:2], preserve_range=True)
    resized_output_thumbnail = resize(output_thumbnail, input_image.shape[:2], preserve_range=True)

    # # FACTOR BASED
    # # Compute color transfer mapping (per channel scaling factor)
    # # Avoid divide-by-zero issues by adding a small epsilon
    # epsilon = 1e-5
    # scaling_factors = (resized_output_thumbnail + epsilon) / (resized_input_thumbnail + epsilon)

    # # Apply the color transfer scaling factor to the original image
    # color_transferred_image = input_image * scaling_factors

    # BIAS BASED
    bias_map = resized_output_thumbnail - resized_input_thumbnail
    color_transferred_image = input_image + bias_map

    # Clip values to valid range and convert back to uint8
    color_transferred_image = np.clip(color_transferred_image, 0, 1)
    # color_transferred_image = (color_transferred_image * 255).astype(np.uint8)

    return color_transferred_image

# Example usage
if __name__ == "__main__":
    import cv2
    # Load images
    input_thumbnail = cv2.imread("apple_thumbnail_in.png")
    output_thumbnail = cv2.imread("apple_thumbnail_out.png")
    input_image = cv2.imread("apple.png")

    # Ensure images are in RGB format
    input_thumbnail = cv2.cvtColor(input_thumbnail, cv2.COLOR_BGR2RGB)
    output_thumbnail = cv2.cvtColor(output_thumbnail, cv2.COLOR_BGR2RGB)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Apply color transfer
    result = apply_color_transfer(input_thumbnail, output_thumbnail, input_image)

    # Save and visualize result
    cv2.imwrite("output_image.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imshow("Color Transferred Image", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
