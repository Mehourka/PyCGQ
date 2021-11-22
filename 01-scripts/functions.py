import gdal
import os
import numpy as np
from skimage import exposure
from WBT.whitebox_tools import WhiteboxTools
import scipy
import csv
#create wbt object necessary for wbtools


resultats= '../03-resultats/'


def gdalOpenDict(x):
    """    Creates gdal dataSets for multiple files stored in a dictionnary """
    for i, j in x.items():
        globals()['%s_ds' %i] = gdal.Open(j)

def extractVars(dic):
    """ 
    extracts variables from Dictionnary
    dictlyr = {'ODM_fn':'C:/GIS/WhiteBT/TIF/ODM_DSM.tif',
                'rODM_fn': 'C:/GIS/WhiteBT/TIF/DSM_Res.tif',
                'MS_fn':'C:/GIS/WhiteBT/TIF/DSMORG.tif'
                }
    gdalOpen(dictlyr)
    extractVars(dictlyr)
    """
    dic2 = dic.copy()
    for i in dic.keys():
        globals()[i] = dic2.pop(i)

def readFiles(filePath, ref=''):
    """   Lecture d'un .tif avec Gdal et numpy  """
    #Lecture de l'Ortho mosaique via GDAL
    file_ds = gdal.Open(filePath)
    nbands = file_ds.RasterCount
    
    if ref!='':
        if ref.RasterXSize != file_ds.RasterXSize or  ref.RasterYSize != file_ds.RasterYSize:
            warped = gdal.GetDriverByName('MEM').Create('',ref.RasterXSize, ref.RasterYSize,nbands,gdal.GDT_Float32)
            gdal.Warp (warped,  file_ds, height= ref.RasterXSize, width = ref.RasterYSize )
            file_ds = warped
    # Transformation des données en numpy array et supperposition
    band_data = []
    for i in range(1,nbands+1):
        band =  file_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(1.0*band)       #les donnés sont ici transofrm2 de int à float
        
    band_data = np.dstack(band_data)        #superposition (stack) des bandes dans un seul fichier
    print(band_data.shape)
    
    # img = ma.masked_equal(exposure.rescale_intensity(band_data),0)            # fonction pour masquer des valeurs en cas de bug (eg: nan) 
    return file_ds, band_data


def writeRaster(layer, out_path, reference_ds, nbands =1, dtype =gdal.GDT_Float32, driver='GTiff', ):
    
    """#Ecriture d'un fichier .tif à partire d'un layer."""
    
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
    
    
# =============================================================================
#                              White Box Tools
# =============================================================================

wbt = WhiteboxTools()   

defaultfunctions= {  # 'nomfonction':[arg1, arg2, ...],
                    'edge_density':[], ## parait bon
                    # 'plan_curvature':[], # parait inutile
                    # 'relative_topographic_position':[250], ## parait moyen, meilleur avec un argument plus grand
                    # 'diff_from_mean_elev':[], ## parait bon
                    # 'elev_percentile':[], ## parait bon
                    # 'elev_relative_to_min_max':[], ## ne fonctionne pas ??
                    # 'find_ridges':[],  # pas tres efficace
                    # 'map_off_terrain_objects':[10],
                    'multiscale_roughness': [5],
                    # 'multiscale_roughness': ['../'+resultats+"wb-features/wbt_scale_", 50],
                    
                    # 'laplacian_filter':[]
                   }
wbt = WhiteboxTools()   


def wbtfunc(fichier_fn,dicfonctions = defaultfunctions):

    """"Calcule des features avec whitebox tools, a partire du {defaultfucntions} dans functions.py"""

    # appel different fonctions stocke dans dictfun sur une image(lien)
    if not os.path.isdir(resultats+'wb-features') : os.mkdir(resultats+'wb-features')
    print(os.getcwd())
    wbt_attribute_list = []
    for funame, args in dicfonctions.items():
        if funame == 'multiscale_roughness':
            getattr(wbt, funame)('../'+fichier_fn,
                                 '../'+resultats+f"wb-features/wbt_{funame}.tif",
                                 '../'+resultats+f"wb-features/wbt_{funame}_scale.tif",
                                 *args)
        else :
            getattr(wbt, funame)('../'+fichier_fn,
                             '../'+resultats+f"wb-features/wbt_{funame}.tif",
                             *args)
        
        assert os.path.exists(os.path.abspath('../'+resultats+f"wb-features/wbt_{funame}.tif")), 'check wbt file paths'
        wbt_attribute_list.append(os.path.abspath('../'+resultats+f"wb-features/wbt_{funame}.tif"))
        os.chdir(os.path.dirname(__file__))
    return wbt_attribute_list






# =============================================================================
#  TEST
# 
#
def IDsegment_features(id,segments,img):
    segment_pixels = img[segments == id]
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            band_stats[3] =0.0
        features += band_stats
    return features
# #  AAA
#
# =============================================================================






def segment_features(segment_pixels):  
    """fonction de calcule des statistiques pour chaque segment """
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            band_stats[3] =0.0
        features += band_stats
    return features




def segments_per_class(segMask, ground_truth):
    """Enumeration des segments (indices) appartenant aux classes des donnés d'entrainement"""
    dict_classes = {}
    classes = np.unique(ground_truth)[1:]
    for klass in classes:
        segments_of_class = segMask[ground_truth == klass]
        dict_classes[klass] = set(segments_of_class)
        print('training segments for class', klass, ':', len(segments_of_class))
    return dict_classes


def intersection_check(segments_per_class):
    """Verifie si un segment represente plus d'une classe et supprime les doublons"""
    intersection = set()
    accum = set()
    for classes_segments in segments_per_class.values():
        intersection |= accum.intersection(classes_segments)
        if len(intersection)!= 0: # Si un segment represente deux classes differentes
            print("**** ATTENTION il y a des doublons dans les donnes d'entrainement ****")
            kye = (list(segments_per_class.keys())[list(segments_per_class.values()).index(classes_segments)])
            segments_per_class[kye] = segments_per_class[kye].difference(intersection)
        accum |= classes_segments
 
    
 
    
def get_train_img(segMask, segments_per_class):
    """Isole les segments d'entrainement"""
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




def split_list(iter,n):
    k, m = divmod(len(iter), n)
    split_data = [iter[i*k + min(i,m)  :  (i+1)*k + min(1+i, m) ]
                  for i in range (n)]
    split_data_order_number = [(i,v) for i, v in enumerate(split_data)]
    
    return split_data_order_number




def get_objects(sub_list ,img ,segments ,q1 ,q2):
    '''
    Calcule des statistics des differents bandes dans img pour chaque segments
    '''
    index = sub_list[0]   # index of sub list
    segment_ids = sub_list[1]   # content of each sub list
    objects = []
    print(f'Job {index} starting\n')
    for id in segment_ids:
        segment_pixels = img[segments == id]
        # print(segment_pixels.shape)
        object_features = segment_features(segment_pixels)
        objects.append(object_features)
    
    q1.put(objects)
    q2.put(index)
    print(f'Job {index} finishing\n')
    

def csv_log(description ,seg_time ,feat_time ,classif_time ,Exec_time ,img_shape ,DSM_fn ,Ortho_fn):
    csv_timelog = resultats+"time-log.csv"
    with open(csv_timelog, 'a', newline='') as f:
        header = ['Description', 'Seg_time', 'Feat_time','Exec_time','Classif_time','Inputs','feat_shape']
        writer = csv.DictWriter(f,header)
        writer.writerow({'Description':description
                          ,'Seg_time': seg_time
                          ,'Feat_time':feat_time
                          ,'Classif_time': classif_time
                          ,'Exec_time': Exec_time
                          ,'feat_shape': img_shape
                          ,'Inputs':f'DSM: {DSM_fn} \n Ortho: {Ortho_fn}'
                        })