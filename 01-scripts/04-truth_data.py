import numpy as np
import geopandas as gpd
import pandas as pd

#Read ShapeFile
gdf = gpd.read_file('C:/CGQ/PyTest/ClipTrain.shp')
class_names = gdf['id'].unique()
print('Les classes sont :', class_names)
class_ids = np.arange(class_names.size) +1
print('class ids', class_ids)
#assigne A class id to each Class and creat CSV file for Ref
df = pd.DataFrame({'label':class_names, 'id':class_ids})
df.to_csv('c:/CGQ/PyTest/Class_lookup.csv')

#Add id to shape file

gdf['idInt'] = gdf['id'].map(dict(zip(class_names, class_ids)))
print('gdf w ids\n\n', gdf.head())

#separer les points en pTrain  et pTest
gdf_train = gdf.sample(frac=0.7)
gdf_test = gdf.drop(gdf_train.index)
print('\n\ngdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test shape', gdf_test.shape)

#creqtion des fichier
gdf_train.to_file('C:/CGQ/PyTest/train.shp')
gdf_test.to_file('C:/CGQ/PyTest/test.shp')


# Rasterize shapefile train data
