#%%
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
from os import path
import glob
import seaborn as sns
import matplotlib as mpl
import json

#%%
mpl.rcParams["font.family"] = "Avenir"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.linewidth"] = 2
#%%
around_ind = 1000  # Centered at indent, size of mask in pixels
ind_size = 400  # Size of square in pixels

num_rel = 50  # number of realizations
num_img_pr = 10  # number of images per realization
size = 224  # size of image required

pattern = "/Users/bhaskar/Documents/Work/hardness/Images/aligned/*.jpg"  # Directory of full-size images
files = [path.basename(x) for x in glob.glob(pattern)]

# %%
im_locs = np.zeros(
    (len(files), num_rel, num_img_pr, 2)
)  # Matrix where locations are stored
# %%
# Creating a mask
img_mask = np.zeros((1920, 2448), np.int32)  # Size of input image

cv2.rectangle(
    img_mask,
    (1240 - int((around_ind) / 2), 1330 - int((around_ind) / 2)),
    (1240 + int((around_ind) / 2), 1330 + int((around_ind) / 2)),
    1,
    -1,
)

rect = [
    [1240, 1330 - int(ind_size / 2)],
    [1240 - int(ind_size / 2), 1330],
    [1240, 1330 + int(ind_size) / 2],
    [1240 + int(ind_size) / 2, 1330],
]
poly = np.array([rect], dtype=np.int32)
cv2.fillPoly(img_mask, poly, 0)
# %% Find sub images
# Only where it is overlapping with the mask
i = 0


while i < len(files):
    j = 0
    while j < num_rel:
        k = 0
        while k < num_img_pr:
            x_c = random.randint(
                1240 - int((around_ind) / 2), 1240 + int((around_ind) / 2)
            )
            y_c = random.randint(
                1330 - int((around_ind) / 2), 1330 + int((around_ind) / 2)
            )

            sub_img = np.zeros((1920, 2448), np.int32)
            cv2.rectangle(
                sub_img,
                (x_c + 1 - int((size) / 2), y_c + 1 - int((size) / 2)),
                (x_c + int((size) / 2), y_c + int((size) / 2)),
                1,
                -1,
            )

            bitwiseAnd = cv2.bitwise_and(sub_img, img_mask)

            if sum(sum(bitwiseAnd)) == size * size:
                im_locs[i, j, k, 0] = x_c
                im_locs[i, j, k, 1] = y_c
                k += 1

        j += 1
    print(i)
    i += 1

#Save so that you don't have to run this again and again
# np.save("img_locs.npy", im_locs)
# %% Plotting the sub-image data

ind_num = 151
realization = 39

ind = cv2.imread(
    "/Users/bhaskar/Documents/Work/hardness/Images/aligned/" + str(ind_num) + ".jpg"
)

im_locs = np.load("datasets/img_locs.npy")
ind_copy = ind.copy()
for sub in range(num_img_pr):
    x_c = int(im_locs[ind_num - 1, realization, sub, 0])
    y_c = int(im_locs[ind_num - 1, realization, sub, 1])
    cv2.rectangle(
        ind,
        (x_c + 1 - int((size) / 2), y_c + 1 - int((size) / 2)),
        (x_c + int((size) / 2), y_c + int((size) / 2)),
        (255, 0, 0),
        5,
    )

    img_sub = ind_copy[
        y_c - int(size / 2) : y_c + int(size / 2),
        x_c - int(size / 2) : x_c + int(size / 2),
    ]

    cv2.imwrite("results/{j:03}_{k:02}.jpg".format(j=ind_num, k=sub), img_sub)

# plt.imshow(ind)
# cv2.imwrite("results/{j:03}_{k:02}.jpg".format(j=ind_num, k=realization), ind)
# %% Indent numbers extract

# indents = np.zeros((len(files), 2))
# HV = np.genfromtxt("hardness_vals.csv", delimiter=",")

# for p in range(indents.shape[0]):
#     indents[p, 0] = int(files[p][:-4])
#     indents[p, 1] = HV[int(files[p][:-4]) - 1, 1]

# indents = indents[indents[:, 0].argsort()]
# # np.save("indent_info.npy", indents)
#%% Plt feature heat map
# plt.figure(figsize=(8,4))
# ax = sns.heatmap(X[146,39,:,:], xticklabels=100, yticklabels=2)
# plt.xlabel('Feature #')
# plt.ylabel('Image #')

# plt.savefig("results/feature_map_ind_151.png", dpi=300, transparent=False, bbox_inches="tight")
# %%

h = json.load(open("history_trained", "r"))
# %%
tr_loss = h["loss"]
val_loss = h["val_loss"]

plt.figure(figsize=(8, 4))
plt.plot(tr_loss, "b", label="Train loss")
plt.plot(val_loss, "r", label="Validation loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
# %%
