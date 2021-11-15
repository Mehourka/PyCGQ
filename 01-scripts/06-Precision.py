import numpy as np
import gdal
import os
import ogr
from sklearn import metrics
import geopandas as gpd
import pandas as pd


imgDir = 'C:/GIS/PyCGQ/02-inputs/'
resultats= 'C:/GIS/PyCGQ/03-resultats/'

ortho_fn = imgDir+'ORTHO-clip.tif'       ; assert os.path.isfile(ortho_fn), f"Chemin '{ortho_fn}' erroné"

ortho_ds = gdal.Open(ortho_fn)


test_fn = imgDir+'shape-files/test.shp'       ; assert os.path.isfile(test_fn), f"Chemin '{test_fn}' erroné"
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driverMem = gdal.GetDriverByName('MEM')
target_ds = driverMem.Create('', ortho_ds.RasterXSize, ortho_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(ortho_ds.GetGeoTransform())
target_ds.SetProjection(ortho_ds.GetProjection())
options = ['ATTRIBUTE=idInt']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()


pred_fn =  resultats+'predictions.tif'
pred_ds = gdal.Open(pred_fn)
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

truth[idx]
np.unique(pred[idx])
cm = metrics.confusion_matrix(truth[idx][pred[idx]<10], pred[idx][pred[idx]<10])

print(cm)
print(cm.diagonal())
print(cm.sum(axis=0))


accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)

test_gdf = gpd.read_file(test_fn)
test_df = pd.DataFrame

test_gdf.keys()
print(test_gdf['id'].unique())


arr = accuracy.reshape(1,len(accuracy))


cm_df = pd.DataFrame(data=cm,index=test_gdf['id'].unique(), columns=test_gdf['id'].unique()) 
cm_df = cm_df.append(pd.DataFrame(data = accuracy.reshape(1,6), index=['Accuracy'], columns= list(cm_df)))
cm_df.to_csv(resultats+'matrice-de-confusion.csv')


gdf = gpd.read_file(test_fn.replace("test","ClipTrain"))
gdf['geometry']
gdf.keys()
