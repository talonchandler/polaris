# `polaris`

`polaris` is a set of tools for reconstructing the orientation of fluorophores
from polarized light microscope data.

## Example run

`polcal | polrecon | polvis`

## macOS Installation

First, clone this repository by runnning

`git clone https://github.com/talonchandler/polaris`

and add it to your `PATH` variable

`export PATH="/path/to/polaris:$PATH"`

Next, install [Anaconda](https://www.anaconda.com/download) and create a
`polaris` environment by running

`conda env create -f environment.yml`

`conda activate polaris`

Add the Anaconda environment to your `PATH` variable

`export PATH="/path/to/anaconda3/bin:$PATH"`

Next, install [Paraview](https://www.paraview.org/download/) and add it to your `PATH` variable

`export PATH="/path/to/paraview/ParaView-5.4.0.app/Contents/bin:$PATH"`

Run `type polcal`, `type python`, and `type pvpython` to check that your
environment is set correctly.

Run `polcal | polrecon | polvis` to run a complete reconstruction. 
