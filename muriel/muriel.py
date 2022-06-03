from os import PathLike

import mrcfile
import starfile
import numpy as np
import einops
import pandas as pd
from pathlib import Path
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
import scipy.misc as smisc

def _subtomo_eulers_to_mat(eulers: np.array):
    # create transformation matrix
    r = R.from_euler('zyz', eulers, degrees=True)
    #matrices = r.as_matrix()
    inv_mat = r.inv()  # takes particle orientations into the tomogram reference frame.
    matrices = inv_mat.as_matrix()
    return matrices

def _relion_star_downgrade(star):
    """
    Downgrade RELION 3.1 STAR file to RELION 3.0 format for Warp
    """
    # Merge optics info into particles dataframe
    data = star['particles'].merge(star['optics'])

    # Get necessary data from 3.1 style star file
    # (RELION 3.0 style expected by warp for particle extraction)
    xyz_headings = [f'rlnCoordinate{axis}' for axis in 'XYZ']
    shift_headings = [f'rlnOrigin{axis}Angst' for axis in 'XYZ']
    euler_headings = [f'rlnAngle{euler}' for euler in ('Rot', 'Tilt', 'Psi')]

    xyz = data[xyz_headings].to_numpy()
    shifts_ang = data[shift_headings].to_numpy()
    pixel_size = data['rlnImagePixelSize'].to_numpy().reshape((-1, 1))
    eulers = data[euler_headings].to_numpy()
    data_out = {}
    data_out['rlnMicrographName'] = data['rlnMicrographName']

    # Get shifts in pixels (RELION 3.0 style)
    shifts_px = shifts_ang / pixel_size

    # update XYZ positions
    xyz_shifted = xyz - shifts_px

    # Create output DataFrame
    df = pd.DataFrame.from_dict(data_out, orient='columns')
    for idx in range(3):
        df[xyz_headings[idx]] = xyz_shifted[:, idx]

    for idx in range(3):
        df[euler_headings[idx]] = eulers[:, idx]
    return df

def read_mrc(filename: PathLike) -> np.ndarray:
    with mrcfile.open(filename) as mrc:
        return mrc.data

def get_pixel_size(filename: PathLike) -> float:
    with mrcfile.open(filename, header_only=True) as mrc:
        return mrc.voxel_size.x

def get_tomo_dims(tomo_name: Path) -> np.ndarray:
    with mrcfile.open(tomo_name, header_only=True) as mrc:
        return np.asarray((mrc.header.nx, mrc.header.ny, mrc.header.nz))

def get_subtomo_information(subtomo_star_file: PathLike, tomogram_name: str):
    """
    This is taken from Alister Burt's relion_star_downgrade program.
    """
    subtomo_md = starfile.read(subtomo_star_file)
    if "optics" in subtomo_md:
        print("Downgrading relion3.1 starfile to 3.0")
        out_df = _relion_star_downgrade(subtomo_md)
        out_df2 = out_df[(out_df['rlnMicrographName'].str.contains(tomogram_name))]
        euler_headings = [f"rlnAngle{axis}" for axis in ("Rot", "Tilt", "Psi")]
        coordinate_headings = [f"rlnCoordinate{axis}" for axis in "XYZ"]
        eulers_particles = out_df2[euler_headings].to_numpy()
        coords_particles = out_df2[coordinate_headings].to_numpy()
    else:
        out_df2 = subtomo_md[(subtomo_md['rlnMicrographName'].str.contains(tomogram_name))]
        euler_headings = [f"rlnAngle{axis}" for axis in ("Rot", "Tilt", "Psi")]
        coordinate_headings = [f"rlnCoordinate{axis}" for axis in "XYZ"]
        eulers_particles = out_df2[euler_headings].to_numpy()
        coords_particles = out_df2[coordinate_headings].to_numpy()
    return coords_particles, eulers_particles


def rescale_volume(volume: np.ndarray, source_angpix: float, target_angpix: float, always_even_dims: bool = True):
    scale_factor = source_angpix / target_angpix
    rescaled_volume = ndi.zoom(volume, scale_factor, order=3)
    if always_even_dims:
        rescaled_volume = force_even_dims(rescaled_volume)
    return rescaled_volume

def crop_volume(volume: np.array, side_length: int):
    volume_center = volume.shape[0] / 2
    half_side_length = side_length / 2
    cvol_min = int(volume_center - half_side_length)
    cvol_max = int(volume_center + half_side_length)
    cropped_volume = volume[
                     cvol_min : cvol_max,
                     cvol_min : cvol_max,
                     cvol_min : cvol_max
                     ]
    return cropped_volume

def force_even_dims(volume: np.ndarray):
    if volume.shape[0] % 2 != 0:
        volume = np.pad(volume, (0, 1), mode='constant')
    return volume


def rotate_volume(volume: np.ndarray, euler_angles: np.ndarray):
    """

    Parameters
    ----------
    volume: 3d array containing a volume
    eulers: (n, 3) array of Euler angles

    Returns
    -------
    rotated_volumes : array of rotated volumes, one per set of Euler angles
    """
    mats = _subtomo_eulers_to_mat(euler_angles)

    # create meshgrid (list of x, y, z coordinates for each point in volume)
    # center coordinates around 0 so that they can be rotated
    zyx = np.meshgrid(*[np.arange(d) - d / 2 for d in volume.shape], indexing='ij')

    # get (nz, ny, nx, 3, 1) array of xyz coordinates
    # [... : References the order of the volume
    # ::-1: References the order of the vectors in the volume]
    xyz = np.stack(zyx, axis=-1)[..., ::-1]
    xyz = einops.rearrange(xyz, 'z y x c -> z y x c 1') # c= coordinate axis

    # apply transformation
    broadcastable_matrices = einops.rearrange(mats, 'n i j -> n 1 1 1 i j')
    transformed_xyz = broadcastable_matrices @ xyz  # (n z y x 3 1)
    transformed_xyz = einops.rearrange(transformed_xyz, 'n z y x c 1 -> n z y x c')

    # re centre coordinates in centre of volume and make zyx
    shifts_xyz = np.asarray(volume.shape)[::-1] / 2
    transformed_xyz += shifts_xyz
    transformed_zyx = transformed_xyz[..., ::-1]

    # sample volume on transformed coordinates
    sampling_coordinates = einops.rearrange(transformed_zyx, 'n z y x c -> c n z y x')
    rotated_volume = ndi.map_coordinates(volume, sampling_coordinates, order=1)
    return rotated_volume


def embed_subvolume_in_volume(volume: np.array,
                              subvolumes: np.array,
                              scoordinates: np.array):
    szyx = np.asarray(subvolumes.shape)[1:] #Box length
    half_sidelength = szyx / 2 #halfbox length
    XYZ_min = scoordinates - half_sidelength #xyz min
    XYZ_max = scoordinates + half_sidelength #xyz max
    XYZ_min = XYZ_min.astype(np.int32)
    XYZ_max = XYZ_max.astype(np.int32)
    ignored_ids =[]
    for idx, (xyz_min, xyz_max) in enumerate(zip(XYZ_min, XYZ_max)):
        if xyz_min[0] <= 0 or \
           xyz_max[0] >= volume.shape[0] or \
           xyz_min[1] <= 0 or xyz_max[1] >= volume.shape[1] or \
           xyz_min[2] <= 0 or xyz_max[2] >= volume.shape[2]:
            # if xy_min[0] < 0:
            #       xy_min[0] = xy_min[0] + MIC.shape[0]
            # if xy_max[0] > MIC.shape[0]:
            #       xy_max[0] = xy_max[0]
            # if xy_min[1] < 0 or xy_max[1] > MIC.shape[1]:
            ignored_ids.append(idx)
        else:
            volume[xyz_min[0]:xyz_max[0],
                   xyz_min[1]:xyz_max[1],
                   xyz_min[2]:xyz_max[2]] += subvolumes[idx]
            # for idx, (xyz_min, xyz_max) in enumerate(zip(XYZ_min, XYZ_max)):
    #     volume[xyz_min[0]:xyz_max[0],
    #            xyz_min[1]:xyz_max[1],
    #            xyz_min[2]:xyz_max[2]] = subvolumes[idx]
    return volume, ignored_ids

