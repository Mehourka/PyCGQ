import gdal
import ogr


#Read Image
Ortho_fn = 'C:/CGQ/PyTest/test1.tif'
Ortho_ds = gdal.Open(Ortho_fn)
#Read train shape file
train_fn = 'C:/CGQ/PyTest/train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

#Create raster in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', Ortho_ds.RasterXSize, Ortho_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(Ortho_ds.GetGeoTransform())
target_ds.SetProjection(Ortho_ds.GetProjectionRef())
# options to set raster between 1 and 6 -> nbr of attributes
options = ['ATTRIBUTE=idInt']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
data = target_ds.GetRasterBand(1).ReadAsArray()

print('min', data.min(),'max',data.max(), 'mean ',data.mean())

