# unclonix
Now only the Python code is ready that compares photos of unique scatterings of nano-/micro-particles in order to confirm/deny the originality of the object. Soon, more updates will be available.

To authenticate a label using a photo, you need to set a threshold similarity index. It is stored in a file.“\Photo\tresh_hold.txt ”
Full initialization (takes time and is performed only when the database is changed):
- In the folders “.\Photo\train\diff” and “.\Photo\train\same”, the user enters various photos of the same and different clusters of the protective label
- The user starts the file. “\Photo\tr.py ”. Statistics on similarity indexes for identical and different clusters are created. It is written by the program to files. “\Photo\hist_diff.txt ”and.“\Photo\hist_same.txt ”
- The user starts the file .\Photo\hist.py . The program builds a histogram of similarity indices and determines the threshold similarity index and records it in “.\Photo\tresh_hold.py ”.

Label Comparison:
- The user starts the file .\Photo\comp.py . The program compares the images “.\Photo\from_base.jpg” and “.\Photo\from_user.jpg ”
A verdict is issued regarding the originality of the label.
