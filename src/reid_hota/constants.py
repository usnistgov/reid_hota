
BOX_FORMAT = "xyxy"  # Choose between (xywh, xyxy)

class AnnotationColumn:
    FRAME="frame"
    OBJECT_ID = "id"
    CLASS_ID = "object_type"

    X1 = "x1"
    Y1 = "y1"
    X2 = "x2"
    Y2 = "y2"

    LAT = "lat"
    LON = "lon"
    ALT = "alt"

    BOX_HASH = "box_hash"
