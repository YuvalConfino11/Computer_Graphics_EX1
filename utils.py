import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        np_img = np_img.astype(np.float32)
        padded_img = np.pad(np_img, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.5)
        grayscale_img_padded = np.dot(padded_img[..., :3], self.gs_weights)
        grayscale_img = grayscale_img_padded[1:-1, 1:-1]
        return grayscale_img.squeeze()
        # raise NotImplementedError("TODO: Implement SeamImage.rgb_to_grayscale")

    # @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        if self.resized_gs.size == 0:
            return np.array([])
        
        np_sample = self.resized_gs.squeeze()

        horDiff = np.diff(np.pad(np_sample, ((0, 0), (0, 1)), 'constant', constant_values=0.5), axis=1)
        verDiff = np.diff(np.pad(np_sample, ((0, 1), (0, 0)), 'constant', constant_values=0.5), axis=0)
        np_sample = np.sqrt(np.square(horDiff) + np.square(verDiff))
        np_sample[np_sample > 1] = 1
        return np_sample

        #raise NotImplementedError("TODO: Implement SeamImage.calc_gradient_magnitude")
        
    def calc_M(self):
        pass
             
    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        if clockwise:
            self.resized_rgb = np.rot90(self.resized_rgb, k=3)
            self.resized_gs = np.rot90(self.resized_gs, k=3)
            self.idx_map_v, self.idx_map_h = np.rot90(self.idx_map_v, k=3), np.rot90(self.idx_map_h, k=3)
        else:
            self.resized_rgb = np.rot90(self.resized_rgb)
            self.resized_gs = np.rot90(self.resized_gs)
            self.idx_map_v, self.idx_map_h = np.rot90(self.idx_map_v), np.rot90(self.idx_map_h)
        self.h, self.w = self.resized_gs.shape

    def init_mats(self):
        pass

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map_v[i, s:] -= 1
            self.idx_map_h[i, s:] -= 1

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)
    
    # @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        # M = np.zeros(self.E.shape, dtype=np.float32)
        # if M.size == 0:  # Check if M is empty
        #     return M
    
        # M[0,:] = self.E[0,:]
        # for i in range(1, self.E.shape[0]):
        #     L_roll = np.roll(M[i-1, :], 1)
        #     R_roll = np.roll(M[i-1, :], -1)
        #     L_roll[0] = np.inf
        #     R_roll[-1] = np.inf
        #     M[i,:] = self.E[i,:] + np.minimum(np.minimum(M[i-1, :], L_roll), R_roll)

        # return M

        M = np.zeros(self.E.shape, dtype = np.float32)
        gsm = np.pad(self.resized_gs, ((1, 1), (1, 1)), mode='constant', constant_values=0.5)

        #calculating cv, cl, cr like we saw in class
        left_shift = np.roll(gsm, -1, axis=1)
        right_shift = np.roll(gsm, 1, axis=1)
        down_shift = np.roll(gsm, 1, axis=0)
        cv = np.abs(left_shift - right_shift)
        cv = cv.squeeze()
        cv = cv[1:-1, 1:-1] 
        cl = np.abs(left_shift - right_shift) + np.abs(right_shift - down_shift)
        cl = cl.squeeze() 
        cl = cl[1:-1, 1:-1]
        cr = np.abs(left_shift - right_shift) + np.abs(left_shift - down_shift)
        cr = cr.squeeze() 
        cr = cr[1:-1, 1:-1]

        #Copying the first row of E
        M[0,:] = self.E[0,:]

        #filling the rest by Forwardlooking Cost formula
        for i in range(1,M.shape[0]):
            l_roll = np.roll(M[i-1,:], 1)
            l_roll[0] = np.inf
            r_roll = np.roll(M[i-1,:], -1)
            r_roll[-1] = np.Inf
            M[i,:] = self.E[i,:] + np.minimum(M[i-1,:] + cv[i] , np.minimum(l_roll + cl[i],r_roll + cr[i]))
            
        return M
        
        #raise NotImplementedError("TODO: Implement SeamImage.calc_M")

    # @NI_decor
    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        for _ in range(num_remove):
            self.init_mats()
            self.remove_seam()
            self.paint_seams()
            self.seam_history= []
    
    def search_seam(self):
        M = self.M
        h, w = M.shape
        mainSeam = np.zeros(h, dtype=int)
        mainSeam[h - 1] = np.argmin(M[h - 1, :])
        for i in range(h - 2, -1, -1):
            j = mainSeam[i + 1]
            if j == 0:
                mainSeam[i] = np.argmin(M[i, :2]) if M[i, :2].size > 0 else 0
            elif j == w - 1:
                mainSeam[i] = j - 1 + (np.argmin(M[i, -2:]) if M[i, -2:].size > 0 else 0)
            else:
                mainSeam[i] = j + (np.argmin(M[i, j - 1:j + 2]) if M[i, j - 1:j + 2].size > 0 else 0) - 1
        return mainSeam
    
    #raise NotImplementedError("TODO: Implement SeamImage.seams_removal")

    def paint_seams(self):
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                if 0 <= s_i < self.w and 0 <= i < self.h:
                    self.cumm_mask[self.idx_map_v[i, s_i], self.idx_map_h[i, s_i]] = False
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1, 0, 0])


    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)
        self.mask = np.ones_like(self.M, dtype=bool)

    # @NI_decor
    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.rotate_mats(clockwise=False)

    # @NI_decor
    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)
        #raise NotImplementedError("TODO: Implement SeamImage.seams_removal_vertical")

    # @NI_decor
    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        h, w = self.M.shape
        seam = []
        # Start from the bottom row and find the minimum energy pixel
        j = np.argmin(self.M[-1])
        seam.append((h-1, j))

        # Move up the rows and find the minimum energy connected pixel
        for i in range(h-2, -1, -1):
            j = max(0, j-1)  # Left boundary
            if j < w - 2:  # Right boundary
                j += np.argmin(self.M[i, j:j+3])  # Find min in the 3x3 window
            seam.append((i, j))

        # Reverse the seam to start from the top
        return seam[::-1]
        # raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam_b")

    def update_ref_mat(self):
        self.idx_map_h = self.idx_map_h[self.mask].reshape(self.h, self.w - 1)
        self.idx_map_v = self.idx_map_v[self.mask].reshape(self.h, self.w - 1)
    # @NI_decor
    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """

        mainSeam = self.search_seam()
        self.seam_history.append(mainSeam)
        self.mask = np.ones_like(self.M, dtype=bool)
        for i in range(len(mainSeam)):
            self.mask[i, mainSeam[i]] = False

        new_mask_3d = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)

        self.resized_rgb = self.resized_rgb[new_mask_3d].reshape(self.h, self.w - 1, 3)
        self.resized_gs = self.rgb_to_grayscale(self.resized_rgb)

        self.update_ref_mat()
        self.w -= 1

        # raise NotImplementedError("TODO: Implement SeamImage.remove_seam")

    # @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")
    
    # @NI_decor
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.rotate_mats(clockwise=True)
        self.seams_addition_vertical(num_add)
        self.rotate_mats(clockwise=False)
        # raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    # @NI_decor
    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """
        self.seams_removal_vertical(num_add)
        self.resized_rgb = np.zeros((self.rgb.shape[0], self.rgb.shape[1] + num_add, 3))

        for r in range(self.seams_rgb.shape[0]):
            i = 0
            for c in range(self.seams_rgb.shape[1]):
                if c + i < self.resized_rgb.shape[1]:
                    self.resized_rgb[r, c + i, :] = self.rgb[r, c, :]
                if self.seams_rgb[r, c, 0] == 1 and self.seams_rgb[r, c, 1] == 0 and self.seams_rgb[r, c, 2] == 0:
                    i += 1
                    if c + i < self.resized_rgb.shape[1]:
                        self.resized_rgb[r, c + i, :] = self.rgb[r, c, :]

        # raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    # @NI_decor
    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, E, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")


class SCWithObjRemoval(VerticalSeamImage):
    def __init__(self, active_masks=['Gemma'], *args, **kwargs):
        import glob
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        self.active_masks = active_masks
        self.obj_masks = {basename(img_path)[:-4]: self.load_image(img_path, format='L') for img_path in glob.glob('images/obj_masks/*')}

        try:
            self.preprocess_masks()
        except KeyError:
            print("TODO (Bonus): Create and add Jurassic's mask")
        
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def preprocess_masks(self):
        """ Mask preprocessing.
            different from images, binary masks are not continous. We have to make sure that every pixel is either 0 or 1.

            Guidelines & hints:
                - for every active mask we need make it binary: {0,1}
        """
        for k in self.active_masks:
            self.obj_masks[k] = (self.obj_masks[k] > 0.5).astype(np.int32)
        # raise NotImplementedError("TODO: Implement SeamImage.preprocess_masks")

    # @NI_decor
    def apply_mask(self):
        """ Applies all active masks on the image
            
            Guidelines & hints:
                - you need to apply the masks on other matrices!
                - think how to force seams to pass through a mask's object..
        """

        for k in self.active_masks:
            self.E = np.where(self.obj_masks[k], 0, self.E + 100)

    def resize_mask(self, mask, new_shape):
        """Resizes a mask to the specified shape using bilinear interpolation."""
        old_height, old_width = mask.shape
        new_height, new_width = new_shape
        resized_mask = np.zeros(new_shape, dtype=mask.dtype)

        for i in range(new_height):
            for j in range(new_width):
                # Find the nearest pixel in the original mask
                nearest_i = round(i * old_height / new_height)
                nearest_j = round(j * old_width / new_width)
                # Clamp the indices to be within the bounds of the original mask
                nearest_i = min(nearest_i, old_height - 1)
                nearest_j = min(nearest_j, old_width - 1)
                # Assign the value of the nearest pixel to the resized mask
                resized_mask[i, j] = mask[nearest_i, nearest_j]

        return resized_mask

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.apply_mask() 
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)
        self.mask = np.ones_like(self.M, dtype=bool)

    def reinit(self, active_masks):
        """ re-initiates instance
        """
        self.__init__(active_masks=active_masks, img_path=self.path)

    def remove_seam(self):
        """ A wrapper for super.remove_seam method. takes care of the masks.
        """
        super().remove_seam()
        for k in self.active_masks:
            self.obj_masks[k] = self.obj_masks[k][self.mask].reshape(self.h, self.w)


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    return np.round(orig_shape * np.array(scale_factors)).astype(int)
    # raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img.reinit()
    seam_img.seams_removal_vertical(np.abs(shapes[0][1] - shapes[1][1]))
    seam_img.seams_removal_horizontal(np.abs(shapes[0][0] - shapes[1][0]))
    
    return seam_img.resized_rgb
    # raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


