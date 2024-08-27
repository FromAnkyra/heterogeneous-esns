from matplotlib import pyplot

f = open("benchmarks/santa-fe-lasers/santafe.txt", "r")
lines = f.readlines()

lines_numeriq = [int(l) for l in lines]

pyplot.plot(lines_numeriq, lw=0.2)
pyplot.show()