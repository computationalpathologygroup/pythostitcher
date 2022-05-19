import os
import numpy as np
import warnings
from .get_resname import get_resname
from .get_filename import get_filename
from .map_tform_low_res import map_tform_low_res
from .packup import packup_edges
from .imfuse import imfuse
from .crop_img import crop_img
from .get_edges import get_edges
from .get_tformed_images import get_tformed_images
from .spatial_ref_object import spatial_ref_object


def map_high_res(base_dir, results_dir, cases, res_num, versions, rev_name,
                 padsizes, res, stitch_identifier):

    case_name = cases["name"]
    slice_name = cases["slice_name"]

    res_name = get_resname(res)
    version = versions["resolution_level"][res_num]

    dirpath = f"{results_dir}/{case_name}/{slice_name}/{res_name}/initialization/"
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    fname_tformAndInitialization = get_filename(dirpath, versions["tform"][res_num], rev_name)

    dirpath_preprocessed = f"{results_dir}/{case_name}/{slice_name}/{res_name}/preprocessed/"
    fname_preprocessed = get_filename(dirpath, versions["preprocessed"][res_num], rev_name)

    if stitch_identifier == "AutoStitch":
        fname = f"{dirpath}{fname_tformAndInitialization}_{stitch_identifier}"
        filepath = fname + ".npy"
        imgpath = fname + ".png"
        fileExists = os.path.isfile(filepath)

    if not fileExists:
        if stitch_identifier == "AutoStitch":

            # These are probably 2 dicts, originally .mat files
            d1 = np.load(dirpath_preprocessed, allow_pickle=False).item()
            d2 = np.load(fname_preprocessed, allow_pickle=False).item()

            images = ["imgA", "imgB", "imgC", "imgD"]
            imdict = dict()

            # Load images from files above
            for im in images:
                if im in d1.keys():
                    imdict[im] = d1[im]
                elif im in d2.keys():
                    imdict[im] = d2[im]

            # Map the transformation from a lower resolution
            initialTform = map_tform_low_res(versions, res_num,
                                             results_dir, case_name, slice_name,
                                             fname_tformAndInitialization)

            # Placeholders for imref2d function
            warnings.warn("Warning: imref2d function is replaced by placeholder")
            rA = spatial_ref_object(np.shape(imdict["imgA"]))
            rB = spatial_ref_object(np.shape(imdict["imgB"]))
            rC = spatial_ref_object(np.shape(imdict["imgC"]))
            rD = spatial_ref_object(np.shape(imdict["imgD"]))

            # Compute transformed images
            imgA_t, rA_t, imgB_t, rB_t, imgC_t, rC_t, imgD_t, rD_t = get_tformed_images(
                imdict["imgA"], imdict["imgB"], imdict["imgC"], imdict["imgD"], initialTform, rA, rB, rC, rD)

            # Again computation with placeholders due to lack of referencing object
            warnings.warn("Warning: imref2d function is replaced by placeholder")
            centerA_xy = [np.mean(rA_t.XWorldLimits), np.mean(rA_t.YWorldLimits)]
            centerB_xy = [np.mean(rB_t.XWorldLimits), np.mean(rB_t.YWorldLimits)]
            centerC_xy = [np.mean(rC_t.XWorldLimits), np.mean(rC_t.YWorldLimits)]
            centerD_xy = [np.mean(rD_t.XWorldLimits), np.mean(rD_t.YWorldLimits)]
            imgCenters = [centerA_xy, centerB_xy, centerC_xy, centerD_xy]

            # Get edges, histograms and intensities
            edge1A_rc, edge2A_rc = get_edges(imgA_t[:, :, 0] > 0, "A", False)
            edge1B_rc, edge2B_rc = get_edges(imgB_t[:, :, 0] > 0, "B", False)
            edge1C_rc, edge2C_rc = get_edges(imgC_t[:, :, 0] > 0, "C", False)
            edge1D_rc, edge2D_rc = get_edges(imgD_t[:, :, 0] > 0, "D", False)
            tformedEdges = packup_edges(edge1A_rc, edge2A_rc, edge1B_rc, edge2B_rc,
                                       edge1C_rc, edge2C_rc, edge1D_rc, edge2D_rc)

            # Transform edges to world coordinates
            edge1A_rc_world_a, edge1A_rc_world_b = rA_t.intrinsic_to_world(edge1A_rc[:, 1], edge1A_rc[:, 0])
            edge1A_rc_world = np.concatenate(edge1A_rc_world_b, edge1A_rc_world_a, axis=1)
            edge2A_rc_world_a, edge2A_rc_world_b = rA_t.intrinsic_to_world(edge2A_rc[:, 1], edge2A_rc[:, 0])
            edge2A_rc_world = np.concatenate(edge2A_rc_world_b, edge2A_rc_world_a, axis=1)

            edge1B_rc_world_a, edge1B_rc_world_b = rB_t.intrinsic_to_world(edge1B_rc[:, 1], edge1B_rc[:, 0])
            edge1B_rc_world = np.concatenate(edge1B_rc_world_b, edge1B_rc_world_a, axis=1)
            edge2B_rc_world_a, edge2B_rc_world_b = rB_t.intrinsic_to_world(edge2B_rc[:, 1], edge2B_rc[:, 0])
            edge2B_rc_world = np.concatenate(edge2B_rc_world_b, edge2B_rc_world_a, axis=1)

            edge1C_rc_world_a, edge1C_rc_world_b = rC_t.intrinsic_to_world(edge1C_rc[:, 1], edge1C_rc[:, 0])
            edge1C_rc_world = np.concatenate(edge1C_rc_world_b, edge1C_rc_world_a, axis=1)
            edge2C_rc_world_a, edge2C_rc_world_b = rC_t.intrinsic_to_world(edge2C_rc[:, 1], edge2C_rc[:, 0])
            edge2C_rc_world = np.concatenate(edge2C_rc_world_b, edge2C_rc_world_a, axis=1)

            edge1D_rc_world_a, edge1D_rc_world_b = rD_t.intrinsic_to_world(edge1D_rc[:, 1], edge1D_rc[:, 0])
            edge1D_rc_world = np.concatenate(edge1D_rc_world_b, edge1D_rc_world_a, axis=1)
            edge2D_rc_world_a, edge2D_rc_world_b = rD_t.intrinsic_to_world(edge2D_rc[:, 1], edge2D_rc[:, 0])
            edge2D_rc_world = np.concatenate(edge2D_rc_world_b, edge2D_rc_world_a, axis=1)

            tformedEdges_rc_world = packup_edges(edge1A_rc_world, edge2A_rc_world, edge1B_rc_world, edge2B_rc_world,
                                                edge1C_rc_world, edge2C_rc_world, edge1D_rc_world, edge2D_rc_world)

            # Create dict with transforms
            tform_ref_objects = dict()
            names = ["rA", "rB_t", "rC_t", "rD_t"]
            ref_objects = [rA_t, rB_t, rC_t, rD_t]

            for name, ref in zip(name, ref_objects):
                tform_ref_objects[str(name)] = ref


            AC, rAC = imfuse(imgA_t, rA_t, imgC_t, rC_t, 'blend', 'none')
            ACB, rACB = imfuse(AC, rAC, imgB_t, rB_t, 'blend', 'none')

            ACBD, rACBD = imfuse(ACB, rACB, imgD_t, rD_t, 'blend', 'none')
            ACBD_cropped, cropVec = crop_img(ACBD, nargout=2)

            minR = cropVec[0]
            maxR = cropVec[1]
            minC = cropVec[2]
            maxC = cropVec[3]

            np.save(imgpath, ACBD_cropped)
            print(f"Saved image in {imgpath}")

    return imgA_t, imgB_t, imgC_t, imgD_t, tform_ref_objects, tformedEdges
