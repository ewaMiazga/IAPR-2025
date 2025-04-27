import numpy as np
import cv2

def rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    return data_red, data_green, data_blue

def hsv_channels(img): 
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    data_h = hsv_img[:, :, 0]
    data_s = hsv_img[:, :, 1]
    data_v = hsv_img[:, :, 2]
    
    return data_h, data_s, data_v

def apply_channels_threshold(img, channels, keep_hsv_thresholds):
    """
    Apply threshold to RGB or HSV input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    channels: str
        Specify the color space to apply thresholding on. Options are 'rgb', 'hsv'.
    keep_hsv_thresholds: dict
        Dictionary containing the lower and upper thresholds for each channel.
        Keys should be 'r', 'g', 'b' for RGB or 'h', 's', 'v' for HSV.
        Values should be lists of tuples [(low1, high1), (low2, high2), ...].

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded binary image.
    """
    if channels == "rgb":
        extracted_channels = rgb_channels(img)
    elif channels == "hsv":
        extracted_channels = hsv_channels(img)
    else:
        raise ValueError("Invalid channels argument. Use 'rgb' or 'hsv'.")

    M, N, _ = img.shape
    img_th = np.zeros((M, N), dtype=np.uint8)

    # Initialize a mask with all True values
    keep_mask = np.ones((M, N), dtype=bool)

    # Iterate over extracted channels and corresponding thresholds
    for channel, (key, thresholds) in zip(extracted_channels, keep_hsv_thresholds.items()):
        channel_mask = np.zeros((M, N), dtype=bool)
        for low, high in thresholds:
            channel_mask = np.logical_or(channel_mask, np.logical_and(channel >= low, channel <= high))
        keep_mask = np.logical_and(keep_mask, channel_mask)

    # Apply the mask to create the thresholded image
    img_th[keep_mask] = 255

    return img_th

