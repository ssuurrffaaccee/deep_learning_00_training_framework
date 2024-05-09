import matplotlib.pyplot as plt
import sys
data =[]
with open(sys.argv[1],"r") as f:
   for line in f:
       data.append(float(line.strip()))
plt.plot(data)
plt.show()
