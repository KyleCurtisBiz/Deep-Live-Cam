import os
import cv2
import numpy as np

# Utility function to support unicode characters in file paths for reading
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

# Utility function to support unicode characters in file paths for writing
def imwrite_unicode(path, img, params=None):
    """Write ``img`` to ``path`` even if the path contains non ASCII characters."""
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
        path = path + ext
    result, encoded_img = cv2.imencode(ext, img, params if params is not None else [])
    if result:
        encoded_img.tofile(path)
    return bool(result)
