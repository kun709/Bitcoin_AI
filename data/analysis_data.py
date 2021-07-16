import pandas as pd
import numpy as np


data = pd.read_csv('larry.csv')
# print(data)

volume = np.asarray(data.iloc[:, 7]) + 1
print('mean  :', volume.mean())
print('std   :', volume.std())
print('min   :', volume.min())
print('max   :', volume.max())
volume = np.clip((0.5 - (1 / (1 + (volume[1:] / volume[:-1])))) * 1.454 + 0.5014, 0, 1)
print('mean  :', volume.mean())
print('std   :', volume.std())
print('mean + 2.58std :', volume.mean() + 2.58 * volume.std())
print('mean - 2.58std :', volume.mean() - 2.58 * volume.std(), '\n')

price = np.asarray(data.iloc[:, 3:7])
mean = np.array([[1.0, 1.0015509, 0.99838033, 1.0]])
std = np.array([[1000, 146.7, 132.8, 124.4]])
price = (price[1:] / price[:-1, [3]] - mean) * std
print('mean :', price.mean(axis=0))
print('std  :', price.std(axis=0))
print('mean + 2.58std :', price.mean(axis=0) + 2.58 * price.std(axis=0))
print('mean - 2.58std :', price.mean(axis=0) - 2.58 * price.std(axis=0))
