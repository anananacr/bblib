# sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
# from maskMakerGUI import pMakePolarisationArray as make_polarization_array_fast
# import geometry_funcs as gf


def correct_polarization(
    x: np.ndarray, y: np.ndarray, dist: float, data: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Correct data for polarisation effect, C built function from https://github.com/galchenm/vdsCsPadMaskMaker/blob/main/SubLocalBG.c#L249
    Acknowledgements: Oleksandr Yefanov, Marina Galchenkova
    Parameters
    ----------
    x: np.ndarray
        x distance coordinates from the direct beam position.
    y: np.ndarray
        y distance coordinates from the direct beam position.
    dist: float
        z distance coordinates of the detector position.
    data: np.ndarray
        Raw data frame in which polarization correction will be applied.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns
    ----------
    corrected_data: np.ndarray
        Corrected data frame for polarization effect.
    pol: np.ndarray
        Polarization array for polarization correction.
    """

    mask = mask.astype(bool)
    mask = ~mask.flatten()
    Int = np.reshape(data.copy(), len(mask))
    pol = mask.copy().astype(np.float32)
    pol = make_polarization_array_fast(
        pol, len(mask), x.flatten(), y.flatten(), dist / Res, 0.99
    )
    mask = ~mask
    pol[np.where(mask == 0)] = 1
    Int = Int / pol
    return Int.reshape(data.shape), pol.reshape(data.shape)
