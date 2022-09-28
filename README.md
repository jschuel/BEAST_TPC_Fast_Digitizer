# BEAST_TPC_Fast_Digitizer

**Note: [ROOT](https://root.cern/install/) is required to run digitize.py and root_pandas (a deprecated package) is required to run produce_primary_tracks.py for electron recoil samples produced in DEGRAD. Instructions for setting up an environment compatible for running all modules follows below the description of the package.**

## Decription
This package contains tools to simulate recoil events in a BEAST TPC. The module assumes the user has already simulated primary recoil tracks using software like [Degrad](https://degrad.web.cern.ch/degrad/) or [SRIM](http://www.srim.org/). **`produce_primary_tracks.py`** will read in these primary recoil tracks and process them into a pandas dataframe to be read into **`digitize.py`**. **`produce_primary_tracks.py`** also has functionality to rotate the primary tracks (following an isotropic angular distribution) to augment event-image sets that may be used for training machine learning classifiers on recoil events.

**`digitize.py`** is the bread and butter of this package. **`digitize.py`** reads in primary tracks and simulates (1) the drifting of charge in a TPC's electric field, (2) the diffusion due to the drift, (3) the amplification of charge, (4) additional diffusion through the amplification device (GEM, micromegas, etc.), (5) [optional] the noise floor of a pixel readout, and (6) the digitization (quantization) of charge into voxels.

![plot](./digi.png)
