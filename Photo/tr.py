import matplotlib.pyplot as plt
from check import *
import os

#warnings.simplefilter("ignore")
a = 1000
b = 500
path1 = "./train/same/"
path2 = "./train/diff/"
f_same = open(".\hist_same.txt", "w")
f_diff = open(".\hist_diff.txt", "w")
for filename1 in os.listdir(path1):
        for filename2 in os.listdir(path1):
                if filename2 == filename1:
                        break
                print(path1+filename1, path1+filename2)
                try :
                        si = check(path1 + filename1, path1 + filename2)
                        print(si)
                        f_same.write(str(si) + " ")
                except:
                        break
                plt.close("all")
f_same.close()
for filename1 in os.listdir(path1):
        for filename2 in os.listdir(path2):
                if filename2 == filename1:
                        break
                print(path2+filename2, path1+filename1)
                try :
                        si = check(path1 + filename1, path2 + filename2)
                        print(si)
                        f_diff.write(str(si) + " ")
                except:
                        f_diff.write("0")
                #print(str(si))
                plt.close("all")
f_diff.close()