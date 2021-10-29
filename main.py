import numpy as np
import numpy.ma as ma
import ogr
import gdal
from skimage import exposure
from skimage.segmentation import quickshift
from skimage.segmentation import slic
import time
import scipy
from sklearn.ensemble import RandomForestClassifier


        #Set File Path
        # Read Raster DataSet
Test_fn = 'C:/CGQ/PyTest/Ortho_clip1.tif'
Ortho_ds = gdal.Open(Test_fn)
nbands = Ortho_ds.RasterCount


# print(type( ma.masked_equal(Ortho_ds.GetRasterBand(1).ReadAsArray() , 0 )  )  )

        #Read bands as arrays and Stack
band_data = []
for i in range(1,nbands):
    band =  Ortho_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
print(band_data.shape)

        #rescale values from 0 to 1
# img = ma.masked_equal(exposure.rescale_intensity(band_data),0)
img =exposure.rescale_intensity(band_data)
# print(img)

#print("this is IMG rescale", img)

        #segmentation
print("segments start")
seg_start = time.time()
segments = quickshift(img, convert2lab=0)
# segments = (img, n_segments=500000, compactness=0.1)

        #create seg raster
segments_fn = 'C:/CGQ/PyTest/segmClipFinal4.tif'
driverTiff = gdal.GetDriverByName('GTiff')
segments_ds = driverTiff.Create(segments_fn, Ortho_ds.RasterXSize, Ortho_ds.RasterYSize,
                                1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(Ortho_ds.GetGeoTransform())
segments_ds.SetProjection(Ortho_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

## Ouvrire le fichier sans re calculer
# segments_ds = gdal.Open('C:/CGQ/PyTest/segmClipFinal.tif')
# segments = segments_ds.GetRasterBand(1).ReadAsArray()
# segments_ds = None

print("segments complete", time.time() - seg_start)

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
segment_ids = np.unique(segments)
objects=[]
object_ids =[]
#loop in ids
print("debut calcule statistiques")
for id in segment_ids:
    segment_pixels = img[segments == id]
    # print(segment_pixels.shape)
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print("fin calcule statistique")
print('created', len(objects), 'objects with', len(objects[0]),'variables in',
      time.time()-objStart, 's')

# from id in segment_ids:
#     segmentPixels =img





#Read train shape file
train_fn = 'C:/CGQ/PyTest/Shp/train.shp'
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

# now we need to get a two dim array for the data in the train raster
ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
classes = np.unique(ground_truth)[1:]
print('class value', classes)

#find which seg belong to which class
segments_per_class = {}
for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print('training segments for class', klass, ':', len(segments_of_class))

#we need to make sure that a set doesnt show in multiple classes
#creating two sets for this
intersection = set()
accum = set()

for classes_segments in segments_per_class.values():
    intersection |= accum.intersection(classes_segments)
    accum |= classes_segments
assert len(intersection) == 0, "segment(s) represent different classes"

###### CLASSIFICATION

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

clfds = driverTiff.Create('C:/CGQ/PyTest/classified4.tif', Ortho_ds.RasterXSize,
                          Ortho_ds.RasterYSize, 1, gdal.GDT_Float32)
clfds.SetGeoTransform(Ortho_ds.GetGeoTransform())
clfds.SetProjection(Ortho_ds.GetProjectionRef())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

target_ds.SetGeoTransform(Ortho_ds.GetGeoTransform())
target_ds.SetProjection(Ortho_ds.GetProjectionRef())