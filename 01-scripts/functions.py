import gdal
import numpy as np
from skimage import exposure
#create wbt object necessary for wbtools





def gdalOpen(x):
    #Creates gdal dataSets for multiple files stored in a dictionnary
    for i, j in x.items():
        globals()['%s_ds' %i] = gdal.Open(j)

def extractVars(dic):
    #extracts variables from Dictionnary
    dic2 = dic.copy()
    for i in dic.keys():
        globals()[i] = dic2.pop(i)
#exemple
"""
dictlyr = {'ODM_fn':'C:/GIS/WhiteBT/TIF/ODM_DSM.tif',
            'rODM_fn': 'C:/GIS/WhiteBT/TIF/DSM_Res.tif',
            'MS_fn':'C:/GIS/WhiteBT/TIF/DSMORG.tif'
            }
gdalOpen(dictlyr)
extractVars(dictlyr)
"""


#Lecture d'un .tif avec Gdal et numpy
def readFiles(filePath):
    #Lecture de l'Ortho mosaique via GDAL
    file_ds = gdal.Open(filePath)
    nbands = file_ds.RasterCount
    
    
    # Transformation des données en numpy array et supperposition
    band_data = []
    for i in range(1,nbands+1):
        band =  file_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(1.0*band)       #les donnés sont ici transofrm2 de int à float
        
    band_data = np.dstack(band_data)        #superposition (stack) des bandes dans un seul fichier
    print(band_data.shape)
    
    img =exposure.rescale_intensity(band_data)          # Rescale des valeurs sur l'interval [0.1]
    # img = ma.masked_equal(exposure.rescale_intensity(band_data),0)            # fonction pour masquer des valeurs en cas de bug (eg: nan) 
    return file_ds, img

#Ecriture d'un fichier .tif à partire d'un layer.
def writeRaster(layer, out_path, reference_ds, nbands =1, dtype =gdal.GDT_Float32, driver='GTiff', ):
    assert nbands == 1, " nbands > 1 : fontion writeRaster dans 'funtions.py' n'est pas encore configurée à prendre plus d'une bande"
    driverTiff = gdal.GetDriverByName(driver)
    segments_ds = driverTiff.Create(out_path, reference_ds.RasterXSize, reference_ds.RasterYSize,
                                    nbands, dtype)
    segments_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    segments_ds.SetProjection(reference_ds.GetProjectionRef())
    if driver=='MEM':
        gdal.RasterizeLayer(segments_ds, [1], layer, options=['ATTRIBUTE=idInt'])
    else:
        segments_ds.GetRasterBand(1).WriteArray(layer)
        
    return segments_ds
    segments_ds = None
    
    
    
    