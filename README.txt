################ Summary ################

MATLAB software for training Gaussian processes and running the 
experiments reported in

Abram L. Friesen and Rajesh P. N. Rao (2011). 
Gaze Following as Goal Inference: A Bayesian Model. 
In L. Carlson, C. Hölscher, & T. Shipley (Eds.), Proceedings of the 33rd 
Annual Conference of the Cognitive Science Society. Boston, MA: 
Cognitive Science Society. July, 2011. 


This software uses code from
(1) The GPML toolbox (http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html)

(2) Marc Deisenroth's ADF code, cited below
Deisenroth, M. P., Huber, M. F., & Hanebeck, U. D. (2009).
Analytic moment-based gaussian process filtering. In Proceedings
of the 26th Annual International Conference on
Machine Learning (pp. 225?232). New York, NY, USA: ACM

(3) The qrot3d MATLAB package (http://www.mathworks.com/matlabcentral/fileexchange/7107-qrot3d-quaternion-rotation)

(4) The arrow MATLAB package (http://www.mathworks.com/matlabcentral/fileexchange/278-arrow-m)

(5) The barweb MATLAB package (http://www.mathworks.com/matlabcentral/fileexchange/10803-barweb--bargraph-with-error-bars-)

(6) The breakplot MATLAB package (http://www.mathworks.com/matlabcentral/fileexchange/21864-breakplot)

(7) The eps2pdf MATLAB package (http://www.mathworks.com/matlabcentral/fileexchange/5782-eps2pdf)


################ License ################
This software is released "as is".  See LICENSE.txt for details. All used software retain their original licenses.


################ Running the code ################

To run the code, you will need to first install the lightspeed toolbox (see
http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/).

The qrot3d library (included in the released software) may also need to be
mex'd (call "mex qrot3d.c" from the qrot3d directory).

Then simply call trainGazeGP and, once trainGazeGP completes, call runGPexp 
to reproduce the experiments in the above paper.

If this software is used for academic purposes, the Friesen & Rao 2011 
paper must be cited.


################ Version history ################
1.0 - initial release