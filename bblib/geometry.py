from om.lib.geometry import TypeDetectorLayoutInformation, TypePixelMaps, TypeDetector, _read_crystfel_geometry_from_text,  _parse_panel_entry, _validate_detector_geometry, _retrieve_layout_info_from_geometry
import collections
import copy
import pprint
import numpy
import math
from om.lib.exceptions import OmGeometryError, OmWrongArrayShape

class VdsGeometryInformation:
    def __init__(
        self,
        *,
        geometry_description: list,
        geometry_format: str,
    ) -> None:
        """
        Detector geometry information from VDS format. Based on OndaMonitor

        This class stores the all the information describing the geometry of an area
        detector. It is initialized with a block of text containing the description of
        the geometry of thr detector (usually the content of a geometry file), and with
        a string specifying the format of the provided information.

        Once the class has been initialized, methods can be invoked to recover
        information about the geometry: lookup-pixel maps, detector's pixel size, etc.

        Arguments:

            geometry_description: a block of text describing the detector's geometry

            geometry_format: a string describing the format of the provided geometry
                description. Currently the following formats are supported:

                * `crystfel`: the geometry format used by the CrystFEL software
                  package.processing of crystallography data. The format is fully
                  documented in CrystFEL's
                  [man pages](http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html)

        Raises:

            OmGeometryError: Raised if the format of the provided geometry information
                is not supported.
        """

        if geometry_format == "crystfel":
            geometry: TypeDetector
            
            geometry, _, __ = _read_crystfel_geometry_from_text(
                text_lines=geometry_description
            )
            self._layout_info: TypeDetectorLayoutInformation = (
                _retrieve_layout_info_from_geometry(geometry=geometry)
            )
            
            self._pixel_maps: TypePixelMaps = _compute_pix_maps(geometry=geometry)

            # Theoretically, the pixel size could be different for every module of the
            # detector. The pixel size of the first module is taken as the pixel size
            # of the whole detector.
            res_first_panel: float = geometry["panels"][
                tuple(geometry["panels"].keys())[0]
            ]["res"]

            # res from crystfel, which is 1/pixel_size
            self._pixel_size: float = 1.0 / res_first_panel

            self._detector_distance_offset: float = geometry["panels"][
                list(geometry["panels"].keys())[0]
            ]["coffset"]
            
        else:
            raise OmGeometryError("Geometry format is not supported.")

    def get_pixel_size(self) -> float:
        """
        Retrieves the size of an area detector pixel.

        This function retrieves information about the size of each pixel making up an
        area detector. All pixels in the detector are assumed to have the same size,
        and have a square shape. The value returned by this function describes the
        length of the side of each pixel.

        Returns:

            Length of the pixel's side in meters.
        """
        return self._pixel_size
    
    def get_pixel_maps(self) -> TypePixelMaps:
        """
        Retrieves pixel maps.

        This function retrieves look-up pixel maps storing coordinate information for
        each pixel of a detector data frame.

        Returns:

            The set of look-up pixel maps.
        """
        return self._pixel_maps

    def get_layout_info(self) -> TypeDetectorLayoutInformation:
        """
        Retrieves detector layout information for the peakfinder8 algorithm.

        This function retrieves information about the internal layout of a detector
        data frame (number and size of ASICs, etc.). This information is needed by the
        [peakfinder8][om.algorithms.crystallography.Peakfinder8PeakDetection] peak
        detection algorithm.

        Returns:

            Internal layout of a detector data frame.
        """
        return self._layout_info

    def get_detector_distance_offset(self) -> float:
        """
        Retrieves detector distance offset information.

        This function retrieves the offset that should be added to the nominal
        detector distance provided by the facility to obtain the real detector distance
        (i.e., the distance between the sample interaction point and the area detector
        where data is recorded. This value is often stored together with the geometry
        information, but if it is not available, the function returns None.

        Returns:

            The detector distance offset in meters, or None if the information is not
            available.
        """
        return self._detector_distance_offset

def _compute_pix_maps(*, geometry: TypeDetector) -> TypePixelMaps:
    # Computes pixel maps from CrystFEL geometry information.
    #for k in geometry["panels"]:
    #    print(geometry["panels"][k])

    max_fs_in_slab: int = 2*( numpy.array(
        [geometry["panels"][k]["orig_max_fs"] for k in geometry["panels"]]
    ).max()+1)-1
    max_ss_in_slab: int = 4*( numpy.array(
        [geometry["panels"][k]["orig_max_ss"] for k in geometry["panels"]]
    ).max()+1)-1

    print(max_fs_in_slab,max_ss_in_slab)
    
    x_map: NDArray[numpy.float_] = numpy.zeros(
        shape=((max_ss_in_slab + 1), (max_fs_in_slab + 1)), dtype=numpy.float32
    )
    y_map: NDArray[numpy.float_] = numpy.zeros(
        shape=((max_ss_in_slab + 1), (max_fs_in_slab + 1)), dtype=numpy.float32
    )
    z_map: NDArray[numpy.float_] = numpy.zeros(
        shape=((max_ss_in_slab + 1), (max_fs_in_slab + 1)), dtype=numpy.float32
    )

    # Iterates over the panels. For each panel, determines the pixel indices, then
    # computes the x,y vectors. Finally, copies the panel pixel maps into the
    # detector-wide pixel maps.


    ##  This part needs to be rewritten for the VDS data format of EuXFEL.
    panel_name: str
    for panel_name in geometry["panels"]:
        panel_number_id = int(geometry["panels"][panel_name]["dim_structure"][1])
        print(panel_number_id, "min_ss", geometry["panels"][panel_name]["orig_min_ss"], "max_ss", geometry["panels"][panel_name]["orig_max_ss"]+1, "min_fs", geometry["panels"][panel_name]["orig_min_fs"], "max_fs", geometry["panels"][panel_name]["orig_max_fs"]+1)
        if "clen" in geometry["panels"][panel_name]:
            first_panel_camera_length: float = geometry["panels"][panel_name]["clen"]
        else:
            first_panel_camera_length = 0.0

        ss_grid: NDArray[numpy.int_]
        fs_grid: NDArray[numpy.int_]
        ss_grid, fs_grid = numpy.meshgrid(
            numpy.arange(
                geometry["panels"][panel_name]["orig_max_ss"]
                - geometry["panels"][panel_name]["orig_min_ss"]
                + 1
            ),
            numpy.arange(
                geometry["panels"][panel_name]["orig_max_fs"]
                - geometry["panels"][panel_name]["orig_min_fs"]
                + 1
            ),
            indexing="ij",
        )
        y_panel: NDArray[numpy.float_] = (
            ss_grid * geometry["panels"][panel_name]["ssy"]
            + fs_grid * geometry["panels"][panel_name]["fsy"]
            + geometry["panels"][panel_name]["cny"]
        )
        x_panel: NDArray[numpy.float_] = (
            ss_grid * geometry["panels"][panel_name]["ssx"]
            + fs_grid * geometry["panels"][panel_name]["fsx"]
            + geometry["panels"][panel_name]["cnx"]
        )

        x_map[
            geometry["panels"][panel_name]["orig_min_ss"] : geometry["panels"][
                panel_name
            ]["orig_max_ss"]
            + 1,
            geometry["panels"][panel_name]["orig_min_fs"] : geometry["panels"][
                panel_name
            ]["orig_max_fs"]
            + 1,
        ] = x_panel
        y_map[
            geometry["panels"][panel_name]["orig_min_ss"] : geometry["panels"][
                panel_name
            ]["orig_max_ss"]
            + 1,
            geometry["panels"][panel_name]["orig_min_fs"] : geometry["panels"][
                panel_name
            ]["orig_max_fs"]
            + 1,
        ] = y_panel
        z_map[
            geometry["panels"][panel_name]["orig_min_ss"] : geometry["panels"][
                panel_name
            ]["orig_max_ss"]
            + 1,
            geometry["panels"][panel_name]["orig_min_fs"] : geometry["panels"][
                panel_name
            ]["orig_max_fs"]
            + 1,
        ] = first_panel_camera_length

    r_map: NDArray[numpy.float_] = numpy.sqrt(numpy.square(x_map) + numpy.square(y_map))
    phi_map: NDArray[numpy.float_] = numpy.arctan2(y_map, x_map)

    return {
        "x": x_map,
        "y": y_map,
        "z": z_map,
        "radius": r_map,
        "phi": phi_map,
    }