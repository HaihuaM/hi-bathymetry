
import os.path as op

DATA_PATH = '../orig_data/GF5_AHSI_E120.98_N31.81_20180715_000983_L10000006312/'
DEPTH_DATA_NAME = "depth_merged.txt"
DEPTH_DATA_PATH = op.join('../orig_data', 'data.depth', DEPTH_DATA_NAME)

LAT_TIFF_FILE = op.join(DATA_PATH, "GF5_AHSI_E120.98_N31.81_20180715_000983_L10000006312.lat.tiff")
LON_TIFF_FILE = op.join(DATA_PATH, "GF5_AHSI_E120.98_N31.81_20180715_000983_L10000006312.lon.tiff")
GEO_TIFF_FILE = op.join(DATA_PATH, "GF5_AHSI_E120.98_N31.81_20180715_000983_L10000006312.geotiff")

OUTPUT_PATH = "../out/"
