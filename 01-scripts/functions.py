import gdal
import os
import numpy as np
from skimage import exposure
from WBT.whitebox_tools import WhiteboxTools
import scipy
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
    
    # img = ma.masked_equal(exposure.rescale_intensity(band_data),0)            # fonction pour masquer des valeurs en cas de bug (eg: nan) 
    return file_ds, band_data

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
    
    
"""                                                                         ******************White Box Tools*******************"""

dictfun = dict()
defaultfunctions= {  # 'nomfonction':[arg1, arg2, ...],

    'edge_density':[], ## parait bon
    # 'plan_curvature':[], # parait inutile
    # 'relative_topographic_position':[250], ## parait moyen, meilleur avec un argument plus grand
    'diff_from_mean_elev':[], ## parait bon
    'elev_percentile':[], ## parait bon
    # 'elev_relative_to_min_max':[], ## ne fonctionne pas ??
    # 'find_ridges':[],  # pas tres efficace
    # 'map_off_terrain_objects':[10],
    'multiscale_roughness': ["../../03-resultats/wb-features/DSM_Res_multiscale_roughness_Scale.tif", 50],
    # 'laplacian_filter':[]

}
wbt = WhiteboxTools()   
def wbtfunc(fichier_fn,dicfonctions = defaultfunctions):
    # appel different fonctions stocke dans dictfun sur une image(lien)
    if not os.path.isdir('../03-resultats/wb-features/') : os.mkdir('../03-resultats/wb-features/')
    print(os.getcwd())
    wbt_attribute_list = []
    for funame, args in dicfonctions.items():
        # if args != '':
        # print(args)
        getattr(wbt, funame)('../'+fichier_fn,
                             f"../../03-resultats/wb-features/wbt_{funame}.tif",
                             *args)
        wbt_attribute_list.append(f"../03-resultats/wb-features/wbt_{funame}.tif")
    os.chdir(os.path.dirname(__file__))
    return wbt_attribute_list



# =============================================================================
#  fonction de calcule des statistiques pour chaque segment
# =============================================================================
def segment_features(segment_pixels):  
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            band_stats[3] =0.0
        features += band_stats
    return features


# =============================================================================
# Enumeration des segments (indices) appartenant aux classes des donnés d'entrainement
# =============================================================================
def segments_per_class(segMask, ground_truth):
    dict_classes = {}
    classes = np.unique(ground_truth)[1:]
    for klass in classes:
        segments_of_class = segMask[ground_truth == klass]
        dict_classes[klass] = set(segments_of_class)
        print('training segments for class', klass, ':', len(segments_of_class))
    return dict_classes

# =============================================================================
# Verifie si un segment represente plus d'une classe et supprime les doublons
# =============================================================================
def intersection_check(segments_per_class):
    intersection = set()
    accum = set()
    for classes_segments in segments_per_class.values():
        intersection |= accum.intersection(classes_segments)
        if len(intersection)!= 0: # Si un segment represente deux classes differentes
            print("**** ATTENTION il y a des doublons dans les donnes d'entrainement ****")
            kye = (list(segments_per_class.keys())[list(segments_per_class.values()).index(classes_segments)])
            segments_per_class[kye] = segments_per_class[kye].difference(intersection)
        accum |= classes_segments

#   Isole les segments d'entrainement
def get_train_img(segMask, segments_per_class):
    train_img = np.copy(segMask)
    threshold = train_img.max() + 1

    for klass in segments_per_class.keys():
        class_label = threshold + klass
        for segment_id in segments_per_class[klass]:
            train_img[train_img == segment_id] = class_label
    # supprime  les segments non identifé et indentifie les segments d'entrainement
    train_img[train_img <= threshold] = 0
    train_img[train_img >= threshold] -= threshold
    return train_img