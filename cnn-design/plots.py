import matplotlib.pyplot


# Graphing results
def plot_input(img):
    fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
    ax0.imshow(img).set_cmap("gray")
    ax0.set_title("Input Image")
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig0)
    

def plot_layer1(l1_feature_map, l1_feature_map_relu, l1_feature_map_relu_pool):
    fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
    ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
    ax1[0, 0].get_xaxis().set_ticks([])
    ax1[0, 0].get_yaxis().set_ticks([])
    ax1[0, 0].set_title("L1-Map1")
    
    ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
    ax1[0, 1].get_xaxis().set_ticks([])
    ax1[0, 1].get_yaxis().set_ticks([])
    ax1[0, 1].set_title("L1-Map2")

    ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax1[1, 0].get_xaxis().set_ticks([])
    ax1[1, 0].get_yaxis().set_ticks([])
    ax1[1, 0].set_title("L1-Map1ReLU")

    ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax1[1, 1].get_xaxis().set_ticks([])
    ax1[1, 1].get_yaxis().set_ticks([])
    ax1[1, 1].set_title("L1-Map2ReLU")

    ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 0].set_title("L1-Map1ReLUPool")

    ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 1].set_title("L1-Map2ReLUPool")

    matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig1)
    

def plot_layer_2(l2_feature_map, l2_feature_map_relu, l2_feature_map_relu_pool):
    fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=3)
    ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
    ax2[0, 0].get_xaxis().set_ticks([])
    ax2[0, 0].get_yaxis().set_ticks([])
    ax2[0, 0].set_title("L2-Map1")

    ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
    ax2[0, 1].get_xaxis().set_ticks([])
    ax2[0, 1].get_yaxis().set_ticks([])
    ax2[0, 1].set_title("L2-Map2")

    ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
    ax2[0, 2].get_xaxis().set_ticks([])
    ax2[0, 2].get_yaxis().set_ticks([])
    ax2[0, 2].set_title("L2-Map3")

    ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax2[1, 0].get_xaxis().set_ticks([])
    ax2[1, 0].get_yaxis().set_ticks([])
    ax2[1, 0].set_title("L2-Map1ReLU")

    ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax2[1, 1].get_xaxis().set_ticks([])
    ax2[1, 1].get_yaxis().set_ticks([])
    ax2[1, 1].set_title("L2-Map2ReLU")

    ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
    ax2[1, 2].get_xaxis().set_ticks([])
    ax2[1, 2].get_yaxis().set_ticks([])
    ax2[1, 2].set_title("L2-Map3ReLU")

    ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax2[2, 0].get_xaxis().set_ticks([])
    ax2[2, 0].get_yaxis().set_ticks([])
    ax2[2, 0].set_title("L2-Map1ReLUPool")

    ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax2[2, 1].get_xaxis().set_ticks([])
    ax2[2, 1].get_yaxis().set_ticks([])
    ax2[2, 1].set_title("L2-Map2ReLUPool")

    ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
    ax2[2, 2].get_xaxis().set_ticks([])
    ax2[2, 2].get_yaxis().set_ticks([])
    ax2[2, 2].set_title("L2-Map3ReLUPool")

    matplotlib.pyplot.savefig("L2.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig2)


def plot_layer_3(l3_feature_map, l3_feature_map_relu, l3_feature_map_relu_pool):
    fig3, ax3 = matplotlib.pyplot.subplots(nrows=1, ncols=3)
    ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
    ax3[0].get_xaxis().set_ticks([])
    ax3[0].get_yaxis().set_ticks([])
    ax3[0].set_title("L3-Map1")

    ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax3[1].get_xaxis().set_ticks([])
    ax3[1].get_yaxis().set_ticks([])
    ax3[1].set_title("L3-Map1ReLU")

    ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax3[2].get_xaxis().set_ticks([])
    ax3[2].get_yaxis().set_ticks([])
    ax3[2].set_title("L3-Map1ReLUPool")

    matplotlib.pyplot.savefig("L3.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig3)