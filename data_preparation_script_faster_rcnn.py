# %%
from sklearn.model_selection import train_test_split
import geopandas as gpd
from utils_file import *
import fiona
import rasterio.mask
# %%
import warnings

warnings.filterwarnings("ignore")
# %%
data_path = '/home/ari/data/ZU/test_new_maskrcnn/'
samples_dir = '/home/ari/data/ZU/test_512'

patch_size = 512

use_AOI = False


# %% md
# Processing for training
# %%
def create_train_ds(input_dir, resulting_ds_dir):
    """
    Preparing data for training by splitting big input images into tiles

        Args:
            input_dir: directory, which contains directories with data that will be used for training
            resulting_ds_dir: directory for resulting splitted images
    """
    raster_path = os.path.join(input_dir, 'Raster')
    GT_path = os.path.join(input_dir, 'Razmetka2')
    AOI_path = os.path.join(input_dir, 'AOI')
    for p in tqdm(os.listdir(raster_path)):
        path = os.path.join(raster_path, p)
        print(path)
        file_3chanel = "three_" + p
        file_name = p.split('.')[0]
        with rasterio.open(os.path.join(path, path)) as rast:
            meta = rast.meta
            out_image = rast.read()
            if use_AOI:
                path_AOI = os.path.join(AOI_path, file_name)
                with fiona.open(path_AOI, "r") as shapefile:
                    shapes = [feature["geometry"] for feature in shapefile]
                    out_image, out_transform = rasterio.mask.mask(rast, shapes)

        orig = gpd.read_file(os.path.join(GT_path, file_name))

        with rasterio.open(file_3chanel, 'w', **meta) as new_dataset:
            new_dataset.write(out_image)

        original_data = orig['geometry'].bounds.apply(inv_affine, 1, file_name=file_3chanel, meta=meta, label='Tree')
        original_data.to_csv(f"{p}_before_proc.csv")
        train_annotations = split_raster(path_to_raster=file_3chanel,
                                         annotations_file=f"{p}_before_proc.csv",
                                         base_dir=resulting_ds_dir,
                                         patch_size=patch_size,
                                         patch_overlap=0.1, allow_empty=False)

        os.remove(file_3chanel)
        os.remove(f"{p}_before_proc.csv")

    final_df = None
    for csv in os.listdir(samples_dir):
        if csv.endswith(".csv"):
            processed = pd.read_csv(os.path.join(resulting_ds_dir, csv)).dropna().reset_index().drop(['index'], axis=1)

            if final_df is None:
                final_df = processed
            else:
                final_df = final_df.append(processed)
    final_df.to_csv(os.path.join(resulting_ds_dir, 'final_df.csv'))


create_train_ds(data_path, samples_dir)
# %%
# perfroming train/test split by images

train_val_test = pd.read_csv(os.path.join(samples_dir, 'final_df.csv'))

train_val_test['label'] = 'Tree'

train_val_paths, test_paths = train_test_split(train_val_test['image_path'].unique(), test_size=0.15, random_state=42)
train_val = train_val_test[train_val_test['image_path'].isin(train_val_paths)]
test = train_val_test[train_val_test['image_path'].isin(test_paths)]
train_val.to_csv(os.path.join(samples_dir, "train_val.csv"))
test.to_csv(os.path.join(samples_dir, 'test.csv'))