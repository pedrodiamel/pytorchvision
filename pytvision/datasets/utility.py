import csv
import errno

import hashlib
import multiprocessing
import os
import os.path
import random
import sys
from io import BytesIO

import cv2
import numpy as np
import requests
import skfmm
import skimage.morphology as morph
import urllib3

from PIL import Image
from scipy import ndimage

from skimage import filters, io, morphology, transform

from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36",
}


def isrgb(image):
    return len(image.shape) == 3 and image.shape[2] == 3


def to_rgb(image):
    # to rgb
    if not isrgb(image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def to_gray(image):
    if isrgb(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def to_channels(image, ch):
    if ch == 1:
        image = to_gray(image)[:, :, np.newaxis]
    elif ch == 3:
        image = to_rgb(image)
    else:
        assert False
    return image


def to_one_hot(x, nc):
    y = np.zeros((nc))
    y[int(x)] = 1.0
    return y


def tolabel(x):
    return np.max(x, axis=0) > 0


def summary(data):
    print(data.shape, data.min(), data.max())


def get_label_mask(mask_img, border_img, seed_ths, threshold, seed_size=8, obj_size=10):
    img_copy = np.copy(mask_img)
    m = img_copy * (1 - border_img)
    img_copy[m <= seed_ths] = 0
    img_copy[m > seed_ths] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = remove_small_objects(img_copy, seed_size).astype(np.uint8)
    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_objects(mask_img, obj_size).astype(np.uint8)
    markers = ndimage.label(img_copy, output=np.uint32)[0]
    labels = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
    return labels


def get_edges(masks):
    edges = np.array([morph.binary_dilation(get_contour(x)) for x in masks])
    return edges


def get_touchs(edges):
    A = np.array([morph.binary_dilation(c, morph.square(3)) for c in edges])
    A = np.sum(A, axis=0) > 1
    I = morph.remove_small_objects(A, 3)
    I = morph.skeletonize(I)
    I = morph.binary_dilation(I, morph.square(3))
    return I


def get_contour(img):
    img = img.astype(np.uint8)
    edge = np.zeros_like(img)
    cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edge, cnt, -1, 1, 1)
    edge = (edge > 0).astype(np.uint8)
    return edge


def get_center(img):
    cent = np.zeros_like(img).astype(np.uint8)
    y, x = ndimage.measurements.center_of_mass(img)
    cv2.circle(cent, (int(x), int(y)), 1, 1, -1)
    cent = (cent > 0).astype(np.uint8)

    cent = np.array([morph.binary_dilation(c) for c in cent])
    cent = tolabel(cent)
    return cent


def get_distance(x):
    return skfmm.distance((x).astype("float32") - 0.5)


def download_images(pack):
    """
    @pack: [ key, url, output ]
    """

    (key, url, out_dir) = pack

    filename = os.path.join(out_dir, "{}.jpg".format(key))
    if os.path.exists(filename):
        print("Image {} already exists. Skipping download.".format(filename))
        return

    try:
        # print('Trying to get %s.' % url)
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        image_data = response.data

        # response = requests.get(d['url'][0], allow_redirects=True, timeout=60, headers=headers)
        # if r.status_code != 200:
        #     print('status code != 200', response.status_code, url )
        #     return
        # image_data = response.content

    except:
        print("Warning: Could not download image {} from {}".format(key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print("Warning: Failed to parse image {} {}".format(key, url))
        return
    try:
        pil_image_rgb = pil_image.convert("RGB")
    except:
        print("Warning: Failed to convert image {} to RGB".format(key))
        return
    try:
        pil_image_rgb.save(filename, format="JPEG", quality=90)
    except:
        print("Warning: Failed to save image {}".format(filename))
        return


def gen_bar_updator(pbar):
    def bar_update(count, block_size, total_size):
        pbar.total = total_size / block_size
        pbar.update(count)

    return bar_update


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)

    print(md5o.hexdigest())
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updator(tqdm()))
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root))
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root),
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def read_image_rgb(self, pathname):
    """
    Load image using pathname
    """

    if os.path.exists(pathname):
        try:
            image = PIL.Image.open(pathname)
            image.load()
        except IOError as e:
            raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message))
    else:
        raise ValueError('"%s" not found' % pathname)

    if image.mode in ["L", "RGB"]:
        # No conversion necessary
        return image
    elif image.mode in ["1"]:
        # Easy conversion to L
        return image.convert("L")
    elif image.mode in ["LA"]:
        # Deal with transparencies
        new = PIL.Image.new("L", image.size, 255)
        new.paste(image, mask=image.convert("RGBA"))
        return new
    elif image.mode in ["CMYK", "YCbCr"]:
        # Easy conversion to RGB
        return image.convert("RGB")
    elif image.mode in ["P", "RGBA"]:
        # Deal with transparencies
        new = PIL.Image.new("RGB", image.size, (255, 255, 255))
        new.paste(image, mask=image.convert("RGBA"))
        return new
    else:
        raise ValueError('Image mode "%s" not supported' % image.mode)

    return image
