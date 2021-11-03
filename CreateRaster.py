

def cr_raster(fn,ref,type):
    
    driverTiff = gdal.GetDriverByName(type)
    segments_ds = driverTiff.Create(fn, ref.RasterXSize, ref.RasterYSize,
                                    1, gdal.GDT_Float32)
    segments_ds.SetGeoTransform(ref.GetGeoTransform())
    segments_ds.SetProjection(ref.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    segments_ds = None