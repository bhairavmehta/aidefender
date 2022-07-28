"""
This script reformat the RP2K dataset to the class images subfolder format, where the base dir
contains sub-directories, which names correspond to class labels, and each sub-directory
contains the corresponding images of this class.
"""
# %%
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

# %%
data_path = Path('/datadrive/aidefender/data/RP2K_categories/categories/')
data_reformatted_path = Path('/datadrive/aidefender/data/RP2K_categories/reformatted/')

categories = [
    'cereals',
    'chocolate',
    'coffee',
    'juice',
    'milk',
    'nuts',
    'oil',
    'soda',
    'spices',
    'tea',
    'vinegar',
    'water',

    'new_labels/alcohol',
    'new_labels/beanpaste',
    'new_labels/beverages',
    'new_labels/bodywash',
    'new_labels/cigarettes',
    'new_labels/cleaners',
    'new_labels/conditioner',
    'new_labels/detergent',
    'new_labels/driedgoods',
    'new_labels/energy_drink',
    'new_labels/facewash',
    'new_labels/hairgel',
    'new_labels/handwash',
    'new_labels/mayonaise',
    'new_labels/moisturizer',
    'new_labels/noodles',
    'new_labels/preserved_foods',
    'new_labels/sauces',
    'new_labels/scented_water',
    'new_labels/seasoning',
    'new_labels/shampoo',
    'new_labels/snacks',
    'new_labels/soap',
    'new_labels/soymilk',
    'new_labels/sunscreen',
    'new_labels/whippedcream',
    'new_labels/yoghurt',

]
# %%
print('Starting')

nb_images_min = 5000
total_images = 0
for cat in tqdm(categories):
    cat_path = data_path.joinpath(cat)
    cat_name = cat_path.name

    cat_reformatted_path = data_reformatted_path.joinpath(cat_name)

    # iterate over files in subdirectories
    cat_images = [file for file in cat_path.rglob('*') if file.is_file()]
    nb_cat_images = len(cat_images)
    print(f'{cat_name}: {nb_cat_images} images', end=', ')

    if nb_cat_images < nb_images_min:
        print('skipping')
    else:
        print('working')

        cat_reformatted_path.mkdir()
        for image_idx, image_file in enumerate(tqdm(cat_images, desc=cat_name)):
            image_name = f'{cat_name}_{image_idx:>05}.jpg'
            try:
                with Image.open(image_file) as image:
                    if image.mode == 'RGB':
                        image.save(cat_reformatted_path.joinpath(image_name))

                        total_images += 1
            except UnidentifiedImageError:
                continue

print(f'Finished: {total_images} images in total')
# %%
