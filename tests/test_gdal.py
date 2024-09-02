def test_gdal_installed():
    from osgeo import gdal

    gdal.VersionInfo()
