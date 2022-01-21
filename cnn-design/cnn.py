import sys
import numpy as np

"""
- no stride
- no padding
"""


def conv_(img, conv_filter):
    """Apply the convolution

    Args:
        img: input image
        conv_filter: filter

    Returns:
        [type]: [description]
    """
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))

    # Looping through the image to apply the convolution operation
    x = filter_size/2.0
    yr = img.shape[0] - filter_size/2.0 + 1
    yc = img.shape[1] - filter_size/2.0 + 1
    for r in np.uint16(np.arange(x, yr)):
        for c in np.uint16(np.arange(x, yc)):
            x_region = np.uint16(np.floor(filter_size/2.0))
            y_region = np.uint16(np.ceil(filter_size/2.0))
            current_region = img[r - x_region:r + y_region,
                                 c - x_region:c + y_region]
            # Element-wise multiplication between the region & filter
            current_result = current_region * conv_filter
            conv_sum = np.sum(current_result)
            result[r, c] = conv_sum

    # Clipping the outliers of the result matrix
    yr = result.shape[0] - np.uint16(filter_size/2.0)
    yc = result.shape[1] - np.uint16(filter_size/2.0)
    final_result = result[np.uint16(x):yr, np.uint16(x):yc]
    return final_result


def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter) > 3:        
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error. The number of channels in both image & filter must match!")
            sys.exit()

    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("Error. The filter must be a square matrix!")
        sys.exit()

    if conv_filter.shape[1] % 2 == 0:
        print("Error. Filter must have an odd size!")
        sys.exit()

    x = img.shape[0] - conv_filter.shape[1] + 1
    y = img.shape[1] - conv_filter.shape[1] + 1
    z = conv_filter.shape[0]
    feature_maps = np.zeros((x, y, z))

    for filter_num in range(z):
        print(f"Filter {filter_num + 1}")
        # Get a filter from the "bank"
        current_filter = conv_filter[filter_num, :]

        # Check if there are multiple channels for the single filter
        if len(current_filter.shape) > 2:
            # Each channel will convolve the image
            conv_map = conv_(img[:, :, 0], current_filter[:, :, 0])
            for ch_num in range(1, current_filter.shape[-1]):
                # Convolving each channel with the image
                # Sum the results
                conv_map = conv_map + conv_(img[:, :, ch_num],
                                            current_filter[:, :, ch_num])
        else:
            conv_map = conv_(img, current_filter)
        feature_maps[:, :, filter_num] = conv_map
    return feature_maps


def relu(feature_map):
    # Preparing the output of the ReLU activation function
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out


def pooling(feature_map, size=2, stride=2):
    # Preparing the output of the pooling operation
    x = np.uint16((feature_map.shape[0] - size + 1) / stride + 1)
    y = np.uint16((feature_map.shape[1] - size + 1) / stride + 1)
    z = feature_map.shape[-1]
    pool_out = np.zeros((x, y, z))
    
    for map_num in range(z):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size, c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out
