from .imfuse import imfuse


def stitch_imfuse(quadrant_A, quadrant_B, quadrant_C, quadrant_D):

    AC, rAC = imfuse(quadrant_A.tform_image,
                     quadrant_C.tform_image,
                     quadrant_A.ref_object,
                     quadrant_C.ref_object,
                     method='blend',
                     options=dict())

    ACB, rACB = imfuse(AC,
                       quadrant_B.tform_image,
                       rAC,
                       quadrant_B.ref_object,
                       method='blend',
                       options=dict())

    ACBD, rACBD = imfuse(ACB,
                         quadrant_D.tform_image,
                         rACB,
                         quadrant_D.ref_object,
                         method='blend',
                         options=dict())

    return ACBD, rACBD
