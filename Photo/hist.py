import matplotlib.pyplot as plt
import numpy as np

f = open("hist_same.txt", "r")
f2 = open("hist_diff.txt", "r")
r_diff = f2.read()
r_same = f.read()
diff = r_diff.split(" ")
same = r_same.split(" ")
#print(diff[:-1])
diff_res = [float(numeric_string) for numeric_string in diff[:-1]]
same_res = [float(numeric_string) for numeric_string in same[:-1]]
print(diff_res)
print(same_res)
plt.hist(diff_res)
plt.hist(same_res)
if(min(same_res) > max(diff_res)):
    #print(min(same_res), max(diff_res))
    tr =str((min(same_res) - max(diff_res))/2 + max(diff_res))
else:
    tr = "Can't determine threshold "

f3 = open("tresh_hold.txt", "w")
f3.write(tr)
f3.close()
plt.show()