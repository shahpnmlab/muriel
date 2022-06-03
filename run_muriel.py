import numpy as np
from pathlib import Path
import typer
import mrcfile

from muriel import get_tomo_dims, \
    rescale_volume, \
    read_mrc, \
    get_pixel_size, \
    rotate_volume, \
    embed_subvolume_in_volume, \
    get_subtomo_information


# AVERAGE_FILE = Path('tests_ali/avg.mrc')
# TOMOGRAM_FILE = Path('tests_ali/Position_19_15.00Apx.mrc')
# TOMOGRAM_NAME = 'TS_01'
# POSE_FILE = Path('tests_ali/VFMatrix.star')
# OUTFILE = Path('tests_ali/test_segmentation.mrc')

def main(AVERAGE_FILE: Path, TOMOGRAM_FILE: Path, TOMOGRAM_NAME: str, POSE_FILE: Path, OUTFILE: Path):
    '''
    DESCRIPTION: A tomogram with the subvolume oriented and placed at the coordinates referenced
    in the star file \n

    PARAMETERS: \n
    AVERAGE_FILE: Path to Subvolume map from Relion or M (eg. /path/to/run_class001.mrc). \n
    TOMOGRAM_FILE: Path to tomogram file (eg. /path/to/TS01_tomo.mrc). \n
    TOMOGRAM_NAME: Name of the Tomogram in the starfile (can match tomogram file name). \n
    POSE_FILE: Path to star file with metadata for subvolume map (eg. /path/to/run_data.star). \n
    OUTFILE: segemented tomogram (eg. /path/to/segmented1.mrc). \n
    '''
    average = read_mrc(AVERAGE_FILE)
    average_pixel_size = get_pixel_size(AVERAGE_FILE)

    tomogram_dimensions = get_tomo_dims(TOMOGRAM_FILE)

    tomogram_pixel_size = get_pixel_size(TOMOGRAM_FILE)

    output_volume = np.zeros(shape=tomogram_dimensions, dtype='float32')

    #cropped_average = crop_volume(average, 220)
    average_rescaled = rescale_volume(volume=average,
                                      source_angpix=average_pixel_size,
                                      target_angpix=tomogram_pixel_size,
                                      always_even_dims=True)

    subvolume_coords, subvolume_eulers = get_subtomo_information(POSE_FILE, TOMOGRAM_NAME)

    rotated_averages = rotate_volume(average_rescaled, euler_angles=subvolume_eulers)

    mapped_back_subvolumes, ignored_subvolume_indices = embed_subvolume_in_volume(output_volume, rotated_averages, subvolume_coords)
    mapped_back_subvolumes = np.swapaxes(mapped_back_subvolumes,2,0)
    with mrcfile.new(OUTFILE, data=mapped_back_subvolumes, overwrite=True) as omrc:
        omrc.voxel_size = tomogram_pixel_size

    with open("ignored_particles.txt",'w') as file:
        for line in ignored_subvolume_indices:
            file.write (f'Particle {line}\n')
    print(f'{len(ignored_subvolume_indices)} particles were ignored as they were too close to the edge.')

if __name__ == '__main__':
    typer.run(main)


