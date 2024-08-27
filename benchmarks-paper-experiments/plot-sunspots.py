from matplotlib import pyplot

f = open("benchmarks/monthly-sunspots.csv", "r")
lines = f.readlines()

lines_numeriq = [float(l.split(',')[1].strip('\n')) for l in lines[1:]]

pyplot.plot(lines_numeriq, lw=0.8)
pyplot.show()