import numpy as np
import gdal
import ogr
import os
from sklearn import metrics


imgDir = 'C:/GIS/PyCGQ/02-inputs/'
resultats= 'C:/GIS/PyCGQ/03-resultats/'

ortho_fn = imgDir+'ORTHO-clip.tif'       ; assert os.path.isfile(ortho_fn), f"Chemin '{ortho_fn}' erroné"
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(ortho_fn)


test_fn = imgDir+'shape-files/test.shp'       ; assert os.path.isfile(test_fn), f"Chemin '{test_fn}' erroné"
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open('C:/GIS/PyCGQ/03-resultats/predictions.tif')
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])

# pixel accuracy
print(cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)