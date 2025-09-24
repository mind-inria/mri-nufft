:orphan:

.. _general_examples:

Examples
========

This is a collection of examples showing how to use MRI-nufft to perform MR image reconstruction.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


GPU Examples
------------

This is a collection of examples showing features of mri-nufft, particularly those that are GPU-accelerated.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples of differents density compensation methods.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_density_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_density.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Density Compensation Routines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we show some tools available to display 3D trajectories. It can be used to understand the k-space sampling patterns, visualize the trajectories, see the sampling times, gradient strengths, slew rates etc. Another key feature is to display the sampling density in k-space, for example to check for k-space holes or irregularities in the learning-based trajectories that would lead to artifacts in the images.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_3d_trajectory_display_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_3d_trajectory_display.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gridded trajectory display</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A small pytorch example to showcase learning k-space sampling patterns. This example showcases the auto-diff capabilities of the NUFFT operator wrt to k-space trajectory in mri-nufft.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_learn_samples_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_learn_samples.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learn Sampling pattern</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A small pytorch example to showcase learning k-space sampling patterns. This example showcases the auto-diff capabilities of the NUFFT operator wrt to k-space trajectory in mri-nufft.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_learn_samples_multicoil_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_learn_samples_multicoil.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learn Sampling pattern for multi-coil MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A small pytorch example to showcase learning k-space sampling patterns. In this example we learn the 2D sampling pattern for a 3D MRI image, assuming straight line readouts. This example showcases the auto-diff capabilities of the NUFFT operator The image resolution is kept small to reduce computation time.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_learn_straight_line_readouts_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_learn_straight_line_readouts.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learn Straight line readout pattern</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to reconstruct volumes using conjugate gradient method.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_cg_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_cg.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reconstruction with conjugate gradient</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This model is a simplified version of the U-Net architecture, which is widely used for image segmentation tasks. This is implemented in the proprietary FASTMRI package [fastmri]_.">

.. only:: html

  .. image:: /generated/autoexamples/GPU/images/thumb/sphx_glr_example_fastMRI_UNet_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_GPU_example_fastMRI_UNet.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Simple UNet model.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>



Operators Examples
------------------

This is a collection of examples showcasing the use of MRI-NUFFT operators for different MR imaging modalities and specificities.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to perform a simple NUFFT.">

.. only:: html

  .. image:: /generated/autoexamples/operators/images/thumb/sphx_glr_example_readme_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_operators_example_readme.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimal example script</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to setup an off-resonance corrected NUFFT operator.">

.. only:: html

  .. image:: /generated/autoexamples/operators/images/thumb/sphx_glr_example_offresonance_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_operators_example_offresonance.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-resonance corrected NUFFT operator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to setup a stacked NUFFT operator.">

.. only:: html

  .. image:: /generated/autoexamples/operators/images/thumb/sphx_glr_example_stacked_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_operators_example_stacked.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stacked NUFFT operator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to setup a subspace NUFFT operator.">

.. only:: html

  .. image:: /generated/autoexamples/operators/images/thumb/sphx_glr_example_subspace_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_operators_example_subspace.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace NUFFT Operator</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>



Trajectories Examples
---------------------

This collection of examples shows how to use MRI-nufft to generate and display k-space sampling trajectories.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A collection of 2D non-Cartesian trajectories with analytical definitions.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_2D_trajectories_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_2D_trajectories.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Trajectories</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A collection of 3D non-Cartesian trajectories with analytical definitions.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_3D_trajectories_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_3D_trajectories.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Trajectories</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An animation to show 2D trajectory customization.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_gif_2D_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_gif_2D.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated 2D trajectories</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An animation to show 3D trajectory customization.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_gif_3D_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_gif_3D.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated 3D trajectories</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example using PyTorch to showcase learning k-space sampling patterns with decimation.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_learn_samples_multires_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_learn_samples_multires.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learning sampling pattern with decimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A collection of sampling densities and density-based non-Cartesian trajectories.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_sampling_densities_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_sampling_densities.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sampling densities</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example to show how to customize trajectory displays.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_display_config_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_display_config.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectory display configuration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A collection of tools to manipulate and develop non-Cartesian trajectories.">

.. only:: html

  .. image:: /generated/autoexamples/trajectories/images/thumb/sphx_glr_example_trajectory_tools_thumb.png
    :alt:

  :ref:`sphx_glr_generated_autoexamples_trajectories_example_trajectory_tools.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectory tools</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /generated/autoexamples/GPU/index.rst
   /generated/autoexamples/operators/index.rst
   /generated/autoexamples/trajectories/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: autoexamples_python.zip </generated/autoexamples/autoexamples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: autoexamples_jupyter.zip </generated/autoexamples/autoexamples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
