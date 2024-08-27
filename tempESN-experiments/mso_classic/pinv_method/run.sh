#!/bin/bash

export PYTHONPATH=~/phd

/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_two.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_three.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_four.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/mso_eight.py $1
# /bin/python3 /home/cw1647/phd/tempESN-experiments/narma.py $1
/bin/python3 /home/cw1647/phd/tempESN-experiments/sunspots.py $1
echo $'\a' 

# but yeah my reasoning is that any further tweaking should really have a strong motivation/rationale for doing so, and the fact that I get decent/competitive results with the two-subreservoir case feels like a good base case to build the rationale upon