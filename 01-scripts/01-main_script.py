import functions 
from WBT.whitebox_tools import WhiteboxTools
import numpy as np
import numpy.ma as ma
import ogr
import gdal
import os
from skimage import exposure
from skimage.segmentation import quickshift
from skimage.segmentation import slic
import time
import scipy
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



# #input
imgDir = 'C:/GIS/PyCGQ/02-inputs/'

Ortho_fn = imgDir+'ORTHO.tif'       ; assert os.path.isfile(Ortho_fn), f"Chemin '{Ortho_fn}' erroné"
DSM_fn = imgDir+'DSM.tif'       ; assert os.path.isfile(Ortho_fn), f"Chemin '{DSM_fn}' erroné"
train_fn = imgDir+'shape-files/train.shp'       ; assert os.path.isfile(train_fn), f"Chemin '{train_fn}' erroné"


#Output
resultats= 'C:/GIS/PyCGQ/03-resultats/'

if not os.path.isdir(resultats) : os.mkdir(resultats)
segments_fn = resultats+'segmentation.tif'
predictions_fn = resultats+'predictions.tif'

#Lecture des donnés de l'ortho
Ortho_ds, img = functions.readFiles(Ortho_fn)

#initialisation de l'objet WhiteboxTools
wbt = WhiteboxTools()


"""                                                    ####################        Segmentation        #####################"""
                                                   
print("segments start")
seg_start = time.time()
 #Ici on choisi la methode de segmentation
                     # # Créer un fichier séparer avec differents methodes de segmentation.
segments = quickshift(img, convert2lab=0)
functions.writeRaster(segments, segments_fn, Ortho_ds)
print("Segmentation complète, {} segments crée en {} secondes".format(np.max(segments),time.time() - seg_start) )

# # Pour ouvrire un fichier de segmentation sans re calculer
        # segments_ds = gdal.Open('C:/CGQ/PyTest/segmClipFinal.tif')
        # segments = segments_ds.GetRasterBand(1).ReadAsArray()
        # segments_ds = None
        
        
"""                                                     ******************* Calcule des Statistiques *************************"""



# fonction de calcule des statistiques pour chaque segment
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

objStart = time.time()

#Selection des segments dans la zone d'interet
allsegments = np.unique(segments)
out_segments = np.unique(segments[img[:,:,1]==0])
segment_ids = np.setdiff1d(allsegments, out_segments)


# Visualisatio des 500 premiers segments (optionnel)
# innersegms = np.zeros_like(band) ; for i in segment_ids[:500]:  innersegms = np.add(innersegms, segments == i)


objects=[]
object_ids =[]
#loop in ids
print(f"debut calcule statistiques pour {np.max(segment_ids)}")
for id in segment_ids:
    segment_pixels = img[segments == id]
    # print(segment_pixels.shape)
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print("fin calcule statistique")
print('creation de ', len(objects), 'objets avec', len(objects[0]),'variables en', time.time()-objStart, 's')


"                                    **********************       Données d'entrainement        *************************************"

#Lectuer des données d'entrainement (shapeFile)
print("lecture des données d'entrainement")
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

#Create raster in memory
target_ds = functions.writeRaster(lyr, '', Ortho_ds, dtype =gdal.GDT_UInt16, driver= 'MEM')
# driver = gdal.GetDriverByName('MEM')
# target_ds = driver.Create('', Ortho_ds.RasterXSize, Ortho_ds.RasterYSize, 1, gdal.GDT_UInt16)
# target_ds.SetGeoTransform(Ortho_ds.GetGeoTransform())
# target_ds.SetProjection(Ortho_ds.GetProjectionRef())
# # options to set raster between 1 and 6 -> nbr of attributes
# options = ['ATTRIBUTE=idInt']
# gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

# now we need to get a two dim array for the data in the train raster
ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
classes = np.unique(ground_truth)[1:]
print('Les valeurs des classes sont : ', classes)

#find which seg belong to which class
segments_per_class = {}
for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print('training segments for class', klass, ':', len(segments_of_class))

#Il faut s'assurer que chaque segment ne represente qu'une seule classe (pas de doublons)
#creating two sets for this
intersection = set()
accum = set()

for classes_segments in segments_per_class.values():
    intersection |= accum.intersection(classes_segments)
    if len(intersection)!= 0: # Si plus d'une classe tombe dans un segment
        print("**** ATTENTION il y a des doublons dans les donnes d'entrainement ****")
        kye = (list(segments_per_class.keys())[list(segments_per_class.values()).index(classes_segments)])
        segments_per_class[kye] = segments_per_class[kye].difference(intersection)
    accum |= classes_segments
# assert len(intersection) == 0, "segment(s) represent different classes"




                                    ########################         CLASSIFICATION         ########################
                
classStart = time.time() 

print("début de la classification")
#create a train image
train_img = np.copy(segments)
#find thershhold to give segments new values ?
threshold = train_img.max() + 1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label
# supprime  les segments non identifé et indentifie les segments d'entrainement
train_img[train_img <= threshold] = 0
train_img[train_img >= threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print("training objects for class",klass, ':', len(class_train_object) )

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('fitting RFC')
predicted = classifier.predict(objects)
print("prediction classifications")

clf = np.copy(segments)
for segment_id, klass in zip(segment_ids,predicted):
    clf[clf == segment_id] = klass

mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0

clf = np.multiply(clf, mask)
clf[clf <0] = -9999.0

clfds = gdal.GetDriverByName('GTiff').Create(predictions_fn, Ortho_ds.RasterXSize,
                          Ortho_ds.RasterYSize, 1, gdal.GDT_Float32)
clfds.SetGeoTransform(Ortho_ds.GetGeoTransform())
clfds.SetProjection(Ortho_ds.GetProjectionRef())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print("Classification effectué en ",time.time()-classStart,"s")
