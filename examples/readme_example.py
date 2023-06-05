
    import matplotlib.pyplot as plt
    import mrinufft
    import numpy as np

    from scipy.dataset import face


    # Create a 2D Radial trajectory for demo
    samples_loc = mrinufft.initialize_2D_radial(Nc=20, Ns=200).reshape((-1, 2))
    # Get a 2D image for the demo
    image = face(gray=True)[:512, :512]

    ## The real deal ##
    # Choose your NUFFT backend (installed independly from the package)
    # And create the associated operator.
    NufftOperator = mrinufft.get_operator("finufft")
    nufft = NufftOperator(samples_loc, shape=(512, 512), density=True, n_coils=1)

    kspace_data = nufft.op(image)  # Image -> Kspace
    image2 = nufft.adj_op(kspace_data) # Kspace -> Image

    # Show the results
    fig, ax =  plt.subplots(2,1)
    ax[0].imshow(image)
    ax[1].imshow(abs(image2))
    plt.show()
