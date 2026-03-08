# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
import cv2
from matplotlib import pyplot as plt
import os


# %%Intermediate feature extraction


# %%
base_model = VGG16(weights="imagenet", include_top=False)
fnames = os.listdir("data_feat_extr")

fnames2 = [x for x in fnames if len(x) == 10]
for fn in fnames2:
    im_temp = cv2.imread("data_feat_extr/" + fn)
    im_cent = im_temp - 157.55
    img_sub = np.expand_dims(im_cent, axis=0)
    # res_dir = os.path.join(script_dir, '/data_feat_extr_copy/im_{}/'.format(fn[:-4]))
    # for lay in range(len(base_model.layers)):
    for lay in range(10):
        mdl = tf.keras.Model(
            inputs=base_model.input, outputs=base_model.layers[lay].output
        )
        feature_maps = mdl.predict(img_sub)

        num_filts = feature_maps.shape[-1]

        cols = 10
        rows = num_filts // cols

        if num_filts % cols != 0:
            rows += 1
        plt.figure(figsize=(7.5, 7.5 * rows / cols))
        # plot the 64 maps in an 8x8 squares
        square = 20
        ix = 1
        for _ in range(num_filts):
            # specify subplot and turn of axis
            ax = plt.subplot(rows, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(
                feature_maps[0, :, :, ix - 1], cmap="inferno"
            )  # 'RdBu','PRGn' 'CMRmap' 'gnuplot2'
            ix += 1

        plt.savefig(
            "data_feat_extr/Results_inferno/im_" + fn[:-4] + "_layer_" + str(lay) + ".png",
            dpi=500,
            bbox_inches="tight",
        )
        # break
        print("Done writing " + "im_" + fn[:-4] + "_layer_" + str(lay) + ".png")
        # if ix > num_filts:
        #     break
# show the figure
# plt.savefig('feat_extr_dummy_bw_inv.png', dpi = 500, bbox_inches = "tight")

# %%
