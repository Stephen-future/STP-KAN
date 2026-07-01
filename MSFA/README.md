# Team: SIP (Spectral Image Processing)
#  Res2-Unet for MSFA-HSI Demosaicing
### The train main code
----train.py
    ## Net architecture
       * The main network (different layers are connected by PixelShuffle and PixelUnShuffle)
-------network_module.py
       * Res2Net and SE_Block used in main network
-------network_module.py
    ## Loss function
-------Loss_function.py
    ## train config
-------trainer.py
### The test main code
----test.py
     (1)mosaic indata/4095   --->[0,1]
     (2)Band separation
     (3)Weighted Bilinear Interpolation
     (4)Network reconstruction
     (5)HS outdata/max()     --->[0,1]
### The submission main code
----prep_submission.py
