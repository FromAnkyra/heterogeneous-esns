#!/bin/bash

export PYTHONPATH=~/phd

# /bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_two.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_three.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_four.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_five.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_six.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_seven.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/mso_classic/ridge_regression/mso_eight.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/ridge_regression/narma.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/ridge_regression/sunspots.py $1

echo $'\a' 
# but yeah my reasoning is that any further tweaking should really have a strong motivation/rationale for doing so, and the fact that I get decent/competitive results with the two-subreservoir case feels like a good base case to build the rationale upon