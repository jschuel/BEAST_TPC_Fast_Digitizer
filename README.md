# BEAST_TPC_Fast_Digitizer
Digitizer module that reads in a simulated primary recoil track (currently tracks simulated in geant4 but there is flexibility for other software), drifts the charge, amplifies the charge, and reads out the charge assuming a pixel ASIC readout. Digitized events are then post processed for direct comparison to measured events.

The current code shifts all events to the origin to save on processing time and allow for easy image creation for passing simulated events into 3D convolutional neural network (3DCNN) classifiers for event selection and directional identification. Since 3DCNNs are translationally invariant, translating events in this way is not an issue.

![plot](./digi.png)
