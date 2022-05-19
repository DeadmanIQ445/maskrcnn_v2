import argparse
from pathlib import Path

import rasterio


def split_channels(_image_path, _dst):
    image = rasterio.open(_image_path)

    meta = image.meta.copy()
    meta['count'] = 1

    rgb = image.read()
    assert len(rgb.shape) == 3 and rgb.shape[0] == 3, f"{rgb.shape} is not valid!"

    r = image.read(1)
    r = r.reshape(1, *r.shape)
    g = image.read(2)
    g = g.reshape(1, *g.shape)
    b = image.read(3)
    b = b.reshape(1, *b.shape)

    with rasterio.open(args.dst / "RED.tif", 'w', **meta) as out:
        out.write(r)

    with rasterio.open(args.dst / "GRN.tif", 'w', **meta) as out:
        out.write(g)

    with rasterio.open(args.dst / "BLU.tif", 'w', **meta) as out:
        out.write(b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split geo image to channels")

    parser.add_argument('--image-path', type=Path, required=True, help='Path to image')
    parser.add_argument('--dst', type=Path, required=True, help="Path to dst folder")
    args = parser.parse_args()

    image_path, dst = args.image_path, args.dst

    assert image_path.is_file()
    assert dst.is_dir()

    split_channels(_image_path=image_path, _dst=dst)
