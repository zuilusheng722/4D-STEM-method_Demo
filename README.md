4D-STEM Methods involve techniques such as centroid, cross-correlation, circle Hough, and radial gradient methods for 
analyzing diffraction patterns obtained from four-dimensional scanning transmission electron microscopy (4D-STEM) datasets. 
These methods utilize a hybrid pixel detector to precisely locate the deflected electron diffraction disks, 
thereby identifying variations in magnetic fields.

In this demonstration, we showcase the application of 4D-STEM methods for locating diffraction disks in the amorphous magnetic 
soft alloy FeSi. The Jupyter notebook "4D_STEM_Centroid.ipynb" provides a detailed walkthrough of each method. For specific 
details on each method, please refer to the accompanying ipynb files. The "circle.py" file contains the source code for all 
methods. You are encouraged to adjust parameters and test the methods with your own data. Ensure that the path to the data 
file "scan_x128_y128.raw" (a 4D-STEM dataset) is correctly specified in the ipynb file, and update the data save paths 
before running the notebooks.

Currently, the supported input file format is .raw. The experimental data was collected in Lorentz STEM mode using 
a Thermo Fischer Themis Z equipped with Lorentz lenses and operating at 300 kV, producing .raw, .tif, and .xml files.

For a comprehensive comparison of various methods for magnetic field measurement using 4D-STEM, refer to the paper 
"Quantitative determination of magnetic fields by four-dimensional scanning transmission electron microscopy: A comparison of various methods" 
by Tao Cheng, Yingze Jia, Luyang Wang, Yangrui Liu, Xudong Pei, Chang Liu, Haifeng Du, Binghui Ge, and Dongsheng Song.
