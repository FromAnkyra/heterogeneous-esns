export PYTHONPATH=~/phd

# /bin/python3 /home/cw1647/phd/UCNC-experiments/extension/mso_two_physical.py
# /bin/python3 /home/cw1647/phd/UCNC-experiments/extension/mso_four_physical.py
# /bin/python3 /home/cw1647/phd/UCNC-experiments/extension/mso_eight_physical.py
# /bin/python3 /home/cw1647/phd/UCNC-experiments/narma-physical.py 50
# /bin/python3 /home/cw1647/phd/UCNC-experiments/narma-fair.py $1
# /bin/python3 /home/cw1647/phd/UCNC-experiments/sunspots-fair.py $1
/bin/python3 /home/cw1647/phd/UCNC-experiments/sunspots-physical.py 50
/bin/python3 /home/cw1647/phd/UCNC-experiments/exp-boxplots.py

echo $'\a' 