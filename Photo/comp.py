from check import *

f = open("tresh_hold.txt", "r")
th = f.read()
th = float(th)
print(th)
f.close()
si = check("./from_user.jpg", "./from_base.jpg")
if si > th:
    print("Original", si)
else:
    print("Fake", si)