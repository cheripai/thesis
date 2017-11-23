#/bin/bash
# Usage: crop_shape.sh <SHAPE_FILE> <INPUT_GEOTIFF> <OUTPUT_GEOTIFF>
gdalwarp -cutline $1 -crop_to_cutline -dstalpha $2 $3
