=========================================
 The trajectory binary file specification
=========================================

The k-space trajectories are transformed to a binary file which is processed by scanner through the arbitrary gradients sequences from NeuroSpin.
This file mainly specifies an arbitrary gradient profile which is played on scanner at gradient raster time rate (10e-6 seconds).

The binary file format is specified as follows:

+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Name           | Type  | Size    | Unit    | Description                                                            |
+================+=======+=========+=========+========================================================================+
| Version        | FLOAT | 1       | n.a.    | file version this new version would be “4.0”                           |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Dimension      | FLOAT | 1       | n.a.    | 2 -> 2D , 3 -> 3D                                                      |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| FOV            | FLOAT | D       | m       | FOV size (x,y,z) : z absent if 2D dimension                            |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Minimum OSF    | FLOAT | 1       | n.a.    | Minimum OS for the trajectory                                          |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Gamma          | FLOAT | 1       | Hz/T    | For Na / MRSI imaging                                                  |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Spokes         | FLOAT | 1       | n.a.    | Number of spokes                                                       |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Samples        | FLOAT | 1       | n.a.    | Number of samples per spoke                                            |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| K-space center | FLOAT | 1       | n.a.    | Relative value in the range [0-1] to define center of spokes           |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| MaxGrad        | FLOAT | 1       | mT/m    | Maximum absolute gradient in all 3 (or 2) directions                   |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| recon_tag      | FLOAT | 1       | n.a.    | Reconstruction tag                                                     |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| timestamp      | FLOAT | 1       | n.a.    | Time stamp when the binary is created                                  |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Empty places   | FLOAT | 9       | n.a.    | Yet unused : Default initialized with 0                                |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| kStarts        | FLOAT | D*Nc    | 1/m     | K-space location start                                                 |
+----------------+-------+---------+---------+------------------------------------------------------------------------+
| Gradient array | FLOAT | D*Nc*Ns | unitary | Gradient trajectory expressed in the range [-1; 1] relative to MaxGrad |
+----------------+-------+---------+---------+------------------------------------------------------------------------+


:mod:`mrinufft.trajectories.io` module helps to convert a trajectory as numpy array to a binary file and vice versa.

All the trajectory FLOAT's are specified with `float32` always.

Note that different versions of the binary file format may have different fields and `mri-nufft` supports IO for `version >= 4.1``
