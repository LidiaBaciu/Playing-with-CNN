import skimage.data
import numpy as np
import cnn
import plots

# Reading the input image
img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

print(img.shape)

# Two filters of size 3x3 will be created.
l1_filter = np.zeros((2, 3, 3))
l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]]])

# Convolve the input image by the created filters

print("**Working with conv layer 1**")
l1_feature_map = cnn.conv(img, l1_filter)

print("**ReLU")
l1_feature_map_relu = cnn.relu(l1_feature_map)

print("**Pooling")
l1_feature_map_relu_pool = cnn.pooling(l1_feature_map_relu, 2, 2)
print("**End of conv layer 1**")

# Second conv layer
l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])

print("**Working with conv layer 2**")
l2_feature_map = cnn.conv(l1_feature_map_relu_pool, l2_filter)

print("**ReLU**")
l2_feature_map_relu = cnn.relu(l2_feature_map)

print("**Pooling**")
l2_feature_map_relu_pool = cnn.pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**")

# Third conv layer
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])

print("**Working with conv layer 3**")
l3_feature_map = cnn.conv(l2_feature_map_relu_pool, l3_filter)

print("**ReLU**")
l3_feature_map_relu = cnn.relu(l3_feature_map)

print("**Pooling**")
l3_feature_map_relu_pool = cnn.pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**")


plots.plot_layer1(l1_feature_map, l1_feature_map_relu, l1_feature_map_relu_pool)
plots.plot_layer_2(l2_feature_map, l2_feature_map_relu, l2_feature_map_relu_pool)
plots.plot_layer_3(l3_feature_map, l3_feature_map_relu, l3_feature_map_relu_pool)