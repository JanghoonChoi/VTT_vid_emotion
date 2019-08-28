import numpy as np
import matplotlib.pyplot as plt
import cv2,time

def get_dtstr(sec=False):
    tst = time.localtime()
    if sec:
        outstr = str(tst.tm_year)[-2:] + str('%02d' % tst.tm_mon) + str('%02d' % tst.tm_mday) + str('%02d' % tst.tm_hour)+ str('%02d' % tst.tm_min)
    else:
        outstr = str(tst.tm_year)[-2:] + str('%02d' % tst.tm_mon) + str('%02d' % tst.tm_mday) + str('%02d' % tst.tm_hour)+ str('%02d' % tst.tm_min)
    return outstr


def imread_to_rgb(path):
    img_in = np.flip(cv2.imread(path, flags=cv2.IMREAD_COLOR), 2)/255.
    return img_in

def imread_to_bw(path):
    img_in = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)/255.
    return img_in

def img_rgb2bw(img):
    img_out = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
    return img_out


def crop_img(I, x, y, w, h, center=False, mfill=False):
    im_h = I.shape[0]
    im_w = I.shape[1]
    
    if center:
        w0 = w // 2;    w1 = w - w0    # w = w0+w1
        h0 = h // 2;    h1 = h - h0    # h = h0+h1

        x_min = x - w0;    x_max = x+w1-1;
        y_min = y - h0;    y_max = y+h1-1;
    else:
        x_min = x;    x_max = x+w-1;
        y_min = y;    y_max = y+h-1;
    
    pad_l = 0;    pad_r = 0;
    pad_u = 0;    pad_d = 0;
    
    # bounds
    if x_min < 0:          pad_l = -x_min;            x_min = 0;
    if x_max > im_w-1:     pad_r = x_max-(im_w-1);    x_max = im_w-1;
    if y_min < 0:          pad_u = -y_min;            y_min = 0;
    if y_max > im_h-1:     pad_d = y_max-(im_h-1);    y_max = im_h-1;

    # crop & append
    J = I[y_min:y_max+1, x_min:x_max+1, :]
    if mfill:
        rsel = np.linspace(0, J.shape[0], 10, endpoint=False, dtype=int)
        csel = np.linspace(0, J.shape[1], 10, endpoint=False, dtype=int)
        fill = np.mean(J[rsel][:,csel], axis=(0,1))
    else:
        fill = (0,0,0)
    J = cv2.copyMakeBorder(J, pad_u,pad_d,pad_l,pad_r, cv2.BORDER_CONSTANT, value=fill)
    return J


def draw_bb_img(img0, x_min,y_min,width,height, color, stroke):
    img = img0.copy()
    img_h = img.shape[0]; img_w = img.shape[1];

    x_rng = np.array(range(width)) + x_min
    y_rng = np.array(range(height))+ y_min
    
    x_rng[x_rng> img_w-1-stroke] = img_w-1-stroke
    y_rng[y_rng> img_h-1-stroke] = img_h-1-stroke
    
    x_max = np.max(x_rng)
    y_max = np.max(y_rng)
    
    img[y_min:y_min+stroke][:, x_rng, :] = color # up
    img[y_max-stroke:y_max][:, x_rng, :] = color # down
    img[:, x_min:x_min+stroke, :][y_rng] = color # left
    img[:, x_max-stroke:x_max, :][y_rng] = color # right
    
    return img


def generate_gaussian_map(map_w,map_h, mean_x,mean_y, sigma):
    sigma = float(sigma)
    out_map = np.zeros([map_h,map_w])

    pos_grid_x = np.tile(range(map_w),(map_h,1))
    pos_grid_y = np.transpose(pos_grid_x)
    pos_grid_x = pos_grid_x.astype(float) - mean_x
    pos_grid_y = pos_grid_y.astype(float) - mean_y

    temp_map = np.exp( -0.5*((np.power(pos_grid_x,2)/sigma**2) + (np.power(pos_grid_y,2)/sigma**2)) )
    out_map = temp_map
    return out_map

def generate_binary_map(map_w, map_h, pos_x, pos_y, radius):
    out_map = - np.ones([map_h, map_w],dtype=float)
    
    col_space = np.array(range(0,map_w,1), dtype=float) #np.linspace(0, map_w-1, map_w)
    lin_space = np.array(range(0,map_h,1), dtype=float) #np.linspace(0, map_h-1, map_h)
    
    col_sqdists = (col_space-pos_x)**2
    lin_sqdists = (lin_space-pos_y)**2
    
    col_sqdists = np.tile(np.expand_dims(col_sqdists, axis=0), (map_h,1))
    lin_sqdists = np.tile(np.expand_dims(lin_sqdists, axis=1), (1,map_w))
    
    all_dists = np.sqrt(col_sqdists+lin_sqdists)
    
    out_map[all_dists<= radius+1.] = 1.
    
    return out_map


def center_of_mass(gmap):
    # weight/height
    gmap_w = gmap.shape[1]; gmap_h = gmap.shape[0]
    
    # return center for all 0 maps
    if gmap.max() <= 0:
        return gmap_h/2, gmap_w/2
    
    # normalize
    gmap -= gmap.min()
    gmap_sum = gmap.sum()
    gmap_wreg = gmap / gmap_sum

    # index matrix
    row_rep = np.tile(np.array(range(gmap_w)), [gmap_h,1])
    col_rep = np.tile(np.array(range(gmap_h)), [gmap_w,1]).transpose()
    # index*weights
    gmap_row = gmap_wreg*row_rep; gmap_col = gmap_wreg*col_rep

    # center corrdinates
    row_mean = np.round( gmap_row.sum() )
    col_mean = np.round( gmap_col.sum() )

    return col_mean,row_mean


def dist_succ(v_pred, v_gt, batch_size):
    maxvals = v_pred.max(axis=1).max(axis=1)
    v_gt_mod = v_gt.copy() + 1.
    
    idxs = list();   gt_idxs = list();
    for b_i in range(batch_size):
        maxpos = np.where(v_pred == maxvals[b_i])[1:3]
        if np.shape(maxpos)[1] > 1:
            maxpos = (np.array([maxpos[0][0]]), np.array([maxpos[1][0]]))
        idxs.append(maxpos)
        gt_idxs.append(center_of_mass(v_gt_mod[b_i]))
        
    idxs = np.array(idxs).reshape([batch_size, 2]).astype(float)
    gt_idxs = np.array(gt_idxs).reshape([batch_size, 2])
    
    dist = np.sum( ( idxs - gt_idxs )**2, axis=1 )
    dist = np.sqrt( dist )
    succ = (dist <= np.sqrt(2.))

    return dist, succ
    

def down2n(x, n):
    # returns input length of x after n-times of pooling/strides of 2
    if n == 1:
        return np.ceil(x/2.)
    else:
        return down2n(np.ceil(x/2.), n-1).astype(int)


# def generate_anchors(bb_input, im_op, anc_changes):
#     # bounding box (x_c, y_c, w, h) to anchor boxes (4, num_anc, map_sz, map_sz)
#     # generate anchor boxes for every position, given bounding box dimensions
#     num_anc = len(anc_changes)
    
#     bb_area  = 64*64 #bb_input[2]*bb_input[3]
#     bb_ratio = 1.00  #bb_input[2]/bb_input[3]
#     map_sz = im_op[1]-im_op[3]+1
#     base_len = float(im_op[0])/float(im_op[1])

#     batch_anc = np.zeros([4, num_anc, map_sz, map_sz])

#     # xy center coordinates
#     coor = (np.arange(map_sz)+(im_op[3]-1)/2.)*base_len
#     coor_x = np.tile(coor.reshape((1,coor.size)), [map_sz,1])
#     coor_y = np.tile(coor.reshape((coor.size,1)), [1,map_sz])
#     coor_xy = np.stack((coor_x,coor_y), axis=0)
#     batch_anc[0:2,:,:,:] = np.tile(coor_xy,[num_anc,1,1,1]).transpose(1,0,2,3)

#     # wh sizes
#     for i in range(num_anc):
#         batch_anc[2,i,:,:] = np.sqrt(bb_area*(bb_ratio*anc_changes[i]))
#         batch_anc[3,i,:,:] = np.sqrt(bb_area/(bb_ratio*anc_changes[i]))

#     return batch_anc


def generate_anchors_fix(bb_unit, im_op, anc_changes, anc_sizes):
    # bounding box (x_c, y_c, w, h) to anchor boxes (4, num_anc, map_sz, map_sz)
    # generate anchor boxes for every position, given bounding box dimensions
    num_cng = len(anc_changes)
    num_szs = len(anc_sizes)
    num_anc = num_cng * num_szs
    # op params
    NUM_DOWNSAMPLE = im_op[2]
    PATCH_SIZE = im_op[0]
    PTMAP_SIZE = down2n(PATCH_SIZE, NUM_DOWNSAMPLE)
    IMAGE_SIZE = im_op[1]
    IMMAP_SIZE = down2n(IMAGE_SIZE, NUM_DOWNSAMPLE)
    HTMAP_SIZE = IMMAP_SIZE - PTMAP_SIZE +1
    
    bb_area  = bb_unit**2 # area of unit box
    map_sz = HTMAP_SIZE
    base_len = float(IMAGE_SIZE)/float(IMMAP_SIZE)

    batch_anc = np.zeros([4, num_anc, map_sz, map_sz])

    # xy center coordinates (same for all types of anchors)
    coor = (np.arange(map_sz)+(PTMAP_SIZE-1)/2.)*base_len
    coor_x = np.tile(coor.reshape((1,coor.size)), [map_sz,1])
    coor_y = np.tile(coor.reshape((coor.size,1)), [1,map_sz])
    coor_xy = np.stack((coor_x,coor_y), axis=0)
    batch_anc[0:2,:,:,:] = np.tile(coor_xy,[num_anc,1,1,1]).transpose(1,0,2,3)

    # wh sizes
    for i in range(num_szs):
        for j in range(num_cng):
            batch_anc[2,num_cng*i+j,:,:] = np.sqrt((bb_area*anc_sizes[i])*(anc_changes[j]))
            batch_anc[3,num_cng*i+j,:,:] = np.sqrt((bb_area*anc_sizes[i])/(anc_changes[j]))

    return batch_anc


def emo2txt(emo):
    #FER: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    emoi = int(emo)
    txt = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    return txt[emoi]
    
    





