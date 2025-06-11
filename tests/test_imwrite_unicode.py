import numpy as np
from modules import imwrite_unicode, imread_unicode

def test_imwrite_unicode(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / 'テスト画像'
    assert imwrite_unicode(str(path), img)
    saved = path.with_suffix('.png')
    assert saved.exists()
    loaded = imread_unicode(str(saved))
    assert loaded.shape == img.shape
