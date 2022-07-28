# %%
import mlflow

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from aidefender.exp.datasets import create_dataset, split_train_val
from aidefender.utils.mlflow import create_art_model
# from aidefender.robustness import robustness_accuracy
from aidefender.robustness import half_distortion
# %%
dataset_class = 'aidefender.exp.datasets.CatsAndDogsDataset'
data_path = '/datadrive/aidefender/data/cats_dogs'

model_path = '../artifacts/models/mlflow_cats_and_dogs_pytorch/'

# dataset_class = 'aidefender.exp.datasets.GroceryDataset'
# data_path = '/datadrive/aidefender/data/grocery/images'

# model_path = '/tmp/aidefender/model/model/'
# %%
nb_intervals = 10
nb_repeats = 3
# %%
dataset = create_dataset(dataset_class, data_path)
dataset_train, dataset_val = split_train_val(dataset, test_size=0.2)
print(f'Dataset [{dataset}]: train {len(dataset_train)}, val: {len(dataset_val)}')
# %%
model_loaded_mlflow = mlflow.pyfunc.load_model(model_path)
model_art = create_art_model(model_loaded_mlflow)
print(f'Model: {type(model_art)}')
# %%
nb_samples_grid = np.linspace(10, len(dataset_val), num=nb_intervals, dtype=np.long)
print(f'Nb sample grid: {nb_samples_grid}')
# %%
scores = []
idx = np.arange(len(dataset_val))

for i in trange(nb_repeats, desc='Repeats'):
    run_scores = []
    for nb_samples in tqdm(nb_samples_grid, desc='Nb sample grid'):
        samples_idx = np.random.choice(idx, size=nb_samples, replace=False)
        images = dataset_val.images[samples_idx]
        labels = dataset_val.labels[samples_idx]

        # score = robustness_accuracy(model_art, images, attack_name='fgsm')
        score = half_distortion(model_art, images, labels, attack_name='pgd')
        run_scores.append(score)

    scores.append(run_scores)

scores = np.array(scores)
# %%
fig, ax = plt.subplots()

mean_scores = np.mean(scores, axis=0)
min_scores = np.min(scores, axis=0)
max_scores = np.max(scores, axis=0)

ax.plot(nb_samples_grid, mean_scores, 'k', color='#CC4F1B')
ax.fill_between(
    nb_samples_grid, min_scores, max_scores, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848'
)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Score')

fig.suptitle('Half distortion score - cats and dogs')

fig.tight_layout()
fig.savefig('/datadrive/aidefender/images/half_distortion_by_nb_samples_cats_and_dogs.pdf')
# %%
