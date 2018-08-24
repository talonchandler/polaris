# `polaris`

`polaris` is a set of tools for reconstructing the spatio-angular density of
fluorophores from data collected with a polarized-light [dual-view inverted selective
plane illumination microscope (diSPIM)](http://dispim.org/).

## Overview

The major functionality of `polaris` is contained in the `Spang` class (short
for spatio-angular density), the `Data` class, and the `MultiMicroscope` class. 

Instances of the `Spang` class contain a spatio-angular density represented as a
4-dimensional array `Spang.f`---three spatial dimensions and one dimension for
spherical harmonic coefficients. `Spang` instances also contain supporting
metadata like the voxel dimensions `Spang.vox_dim`. The `Spang` class contains
methods for calculating summary statistics `Spang.calc_stats`, visualization
(`Spang.visualize`, `Spang.save_summary`), saving to file `Spang.save_tiff`, and
reading from file `Spang.read_tiff`.

Instances of the `Data` class contain 5-dimensional data sets collected by the
microscope `Data.g`---three spatial dimensions, one polarizer dimension, and one
view dimension. `Data` instances also contain metadata like the polarizer
orientations `Data.pols` and the viewing directions `Data.det_optical_axes`. The
`Data` class contains methods for visualization `Data.save_mips`, saving to file
`Data.save_tiff`, and reading from file `Data.read_tiff`.

Finally, the `MultiMicroscope` class implements forward and inverse mappings
between `Spang` objects and `Data` objects. We can perform a forward simulation
with

    data.g = micro.fwd(spang.f)

Similarly, a pseudoinverse solution can be found with

    spang.f = micro.pinv(data.g)

See the `example` scripts for complete forward and inverse simulations.

## Getting started

These instructions have only been tested on macOS, but near variants should work
on all platforms.

Is anaconda installed? If not install it through `brew` or from the [anaconda
webpage](https://www.anaconda.com/download/).

Clone a copy of polaris.

    git clone https://github.com/talonchandler/polaris.git

Create an anaconda environment for polaris. This will take ~5 minutes. 

    cd polaris
    conda env create -f environment.yml

Activate the polaris environment. You will need to activate this environment 
every time you want to run polaris. 

    conda activate polaris

Install polaris locally so that you can access it from anywhere. 

    pip install -e ./

Alternatively, you can modify your `$PYTHONPATH` variable. You will only need to
do this once.

Run the example scripts.

    cd examples
    python demo-synthetic-helix.py
    python demo-data-guv.py