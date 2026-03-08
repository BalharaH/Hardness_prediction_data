#%%
import numpy as np
import cv2

#%%
im_locs = np.load("datasets/img_locs.npy")
ind_data = np.load("datasets/indent_info.npy")
# %%
sum_px = 0
size = 224
for i in range(im_locs.shape[0]):
    
    im_temp = cv2.imread(
                "/Users/bhaskar/Documents/Work/hardness/Images/aligned/"
                + str(int(ind_data[i, 0]))
                + ".jpg")
    for j in range(im_locs.shape[1]):
        for k in range(im_locs.shape[2]):
            img_sub = im_temp[int(im_locs[i,j,k,1]) - int(size / 2) : int(im_locs[i,j,k,1]) + int(size / 2),int(im_locs[i,j,k,0]) - int(size / 2) : int(im_locs[i,j,k,0]) + int(size / 2)]
            sum_px += np.mean(img_sub)
            print([i,j,k])
#%%
sum_px = 0
size = 224
for i in range(1):
    im_temp = cv2.imread(
                "/Users/bhaskar/Documents/Work/hardness/Images/aligned/"
                + str(int(ind_data[i, 0]))
                + ".jpg")
    for j in range(1):
        for k in range(5):
            img_sub = im_temp[int(im_locs[i,j,k,1]) - int(size / 2) : int(im_locs[i,j,k,1]) + int(size / 2),int(im_locs[i,j,k,0]) - int(size / 2) : int(im_locs[i,j,k,0]) + int(size / 2)]
            sum_px += np.mean(im_temp)
# %%
# Mean of all pixels is 157.55
# Subtract this from all input image sets
