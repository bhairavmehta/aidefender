# %%
import os
import numpy as np

from aidefender.exp.datasets import create_dataset
# %%
dataset_class = 'aidefender.exp.datasets.RP2KDataset'
data_path = '/datadrive/aidefender/data/RP2K_categories/reformatted/'

output_path = os.path.join(data_path, 'data.npz')
# %%
dataset = create_dataset(dataset_class, data_path)
# %%
print(f'Images: {dataset.images.shape}')
print(f'Labels: {dataset.labels.shape}')
# %%
np.savez_compressed(output_path, images=dataset.images, labels=dataset.labels)
print(f'Saved: {output_path}')
# %%
