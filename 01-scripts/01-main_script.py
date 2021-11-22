import functions 
import numpy as np
import gdal
import ogr
import os
from skimage import exposure
from skimage.segmentation import quickshift
import time
import scipy
from sklearn.ensemble import RandomForestClassifier
import csv
import multiprocessing as mp

# from WBT.whitebox_tools import WhiteboxTools
# import numpy.ma as ma
# from skimage.segmentation import slic
# import matplotlib.pyplot as plt
import pickle




def main():
    start_time = time.perf_counter()
    
    # #input
    os.chdir(os.path.dirname(__file__))
    
    imgDir = '../02-inputs/'
    Ortho_fn = imgDir+'ORTHO.tif'       ; assert os.path.isfile(Ortho_fn), f"Chemin '{Ortho_fn}' erroné"
    DSM_fn = imgDir+'DSM.tif'       ; assert os.path.isfile(Ortho_fn), f"Chemin '{DSM_fn}' erroné"
    train_fn = imgDir+'shape-files/train.shp'       ; assert os.path.isfile(train_fn), f"Chemin '{train_fn}' erroné"
    
    
    #Output
    resultats= '../03-resultats/'
    
    if not os.path.isdir(resultats) : os.mkdir(resultats)
    segments_fn = resultats+'segmentation.tif'
    predictions_fn = resultats+'predictions.tif'
    
    #Lecture des donnés de l'ortho
    Ortho_ds, bdata_ortho = functions.readFiles(Ortho_fn)
    DSM_ds, bdata_dsm =  functions.readFiles(DSM_fn, Ortho_ds)
    
    
    #Calcule des features avec whiteboxtools (choix des feautres dans le module functions.py)
    wbt_list = functions.wbtfunc(DSM_fn)
    
    feature_bands = bdata_dsm
    for i in wbt_list:
        dataset, bands = functions.readFiles(i, Ortho_ds)
        feature_bands= np.concatenate((feature_bands,bands),axis=2)
    
    img = exposure.rescale_intensity(bdata_ortho)
    
    
    #commentaire pour le log
    csv_timelog = resultats+"time-log.csv"
    # =============================================================================
    #                                    Segmentation   
    # =============================================================================                                                                 
    print("segments start")
    seg_start = time.perf_counter()
     #Ici on choisi la methode de segmentation
                         # # Créer un fichier séparer avec differents methodes de segmentation.
    # segments = quickshift(img, convert2lab=0)    
    # functions.writeRaster(segments, segments_fn, Ortho_ds)
    
    # # Pour ouvrire un fichier de segmentation sans re calculer
    segments_ds = gdal.Open(segments_fn)
    segments = segments_ds.GetRasterBand(1).ReadAsArray()
    segments_ds = None
    seg_time = time.perf_counter()-seg_start
    print(f"Segmentation complète, {np.max(segments)} segments crée en {seg_time} secondes" )
      
    # =============================================================================
    #                              Calcule des Statistiques 
    # =============================================================================
    # img = np.concatenate((bdata_ortho,bdata_dsm),axis=2)
    img = np.concatenate((bdata_ortho, feature_bands),axis=2)
    obj_start = time.perf_counter()
    #Selection des segments dans la zone d'interet
    allsegments = np.unique(segments)
    out_segments = np.unique(segments[img[:,:,1]==0])
    segment_ids = np.setdiff1d(allsegments, out_segments)


    # objects = []
    # object_ids = []
    # # loop in ids
    # print(f"debut calcule statistiques pour {np.shape(segment_ids)}")
    # for id in segment_ids:
    #     segment_pixels = img[segments == id]
    #     # print(segment_pixels.shape)
    #     object_features = functions.segment_features(segment_pixels)
    #     objects.append(object_features)
    #     object_ids.append(id)



    # =============================================================================
    #                        MULTIPROCESS Calcule des Statistiques MULTIPROCESS
    # =============================================================================
    nCpu = mp.cpu_count()
    split_segment_ids = functions.split_list(segment_ids, nCpu)
    # initial Queue which will be used to save our output
    # you can initiate as many Queue as you wantqout1 = mp.Queue()
    qout1 = mp.Queue()
    qout2 = mp.Queue()
    
    
    processes = [mp.Process(target = functions.get_objects, args=(sub_list ,img ,segments ,qout1 ,qout2)) for sub_list in split_segment_ids]
    for p in processes:
        p.daemon=True
        p.start()
    
    objects = sum([qout1.get() for p in processes],[])
    for p in processes:
        p.join()
        p.close()
    
    
    print("fin calcule statistique")
    
    feat_time = time.perf_counter()-obj_start
    print(f'creation de {len(objects)}objets avec {len(objects[0])} variables en{feat_time} seconds')
    
    
    
    
    # =============================================================================
    #                           Données d'entrainement 
    # =============================================================================
    
    
    #Read train data (shapeFile)
    print("lecture des données d'entrainement")
    train_ds = ogr.Open(train_fn)
    lyr = train_ds.GetLayer()
    #Rasterize the vector data in memory
    target_ds = functions.writeRaster(lyr, '', Ortho_ds, dtype =gdal.GDT_UInt16, driver= 'MEM')
    # Read an NpArray
    ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
    classes = np.unique(ground_truth)[1:]
    print('Les valeurs des classes sont : ', classes)
    
    #find which seg belong to which class
    segments_per_class = functions.segments_per_class(segments,ground_truth)
    
    #Il faut s'assurer que chaque segment ne represente qu'une seule classe (pas de doublons)
    functions.intersection_check(segments_per_class)
    #create a train image
    train_img = np.copy(segments)
    #find thershhold to give segments new values ?
    # train_img = functions.get_train_img(segments, segments_per_class)
    
    # train_img = np.copy(segments)
    # threshold = train_img.max() + 1
    
    # for klass in classes:
    #     class_label = threshold + klass
    #     for segment_id in segments_per_class[klass]:
    #         train_img[train_img == segment_id] = class_label
    # # supprime  les segments non identifé et indentifie les segments d'entrainement
    # train_img[train_img <= threshold] = 0
    # train_img[train_img >= threshold] -= threshold
    
    # =============================================================================
    #                               CLASSIFICATION
    # =============================================================================
                    
    class_start = time.perf_counter() 
    
    print("début de la classification")
    training_objects = []
    training_labels = []
    
    for klass in classes:
        class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
        training_labels += [klass] * len(class_train_object)
        training_objects += class_train_object
        print("training objects for class",klass, ':', len(class_train_object) )
    
    classifier = RandomForestClassifier(n_jobs=-1)
    fit_rfc = classifier.fit(training_objects, training_labels)
    
    #
    #saving classifier
    save_rfc = open(imgDir+'fitRfc.pickle','wb')
    pickle.dump(fit_rfc, save_rfc)
    save_rfc.close()
    
    
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
    
    
    classif_time = time.perf_counter()-class_start
    print(f"Classification effectué en {classif_time} s")
    
    
    Exec_time = time.perf_counter()- start_time
    functions.csv_log('Full Img - MultiProcessing 100%'
                      ,seg_time ,feat_time ,classif_time ,Exec_time ,img.shape ,DSM_fn ,Ortho_fn)
        
        
if __name__ == "__main__":
    main()