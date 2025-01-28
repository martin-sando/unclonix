To run project:
1) Create directories unclonix/input and unclonix/output. Add label photos in input.
2) Execute key_generation_sandbox. Use no arguments for default. Supported arguments are: --prefix(only execute photos starting with prefix), --mask(only containing given mask), --reverse(if you want to run then in reverse order), bot(to use bot handler)
3) Wait until finished (may take long). Hashes will be in the console output and in output/hashes.txt file (right now, there are more than 1 hash, but none are working properly)
4) You may also see debug output in output directory

Executing order:
1) In key_generation_sandbox, all photos are executed sequentially (or bot is turned on, that is special case though)
2) If photo name is valid (for prefix and mask), it is processed in 2 phases: first one in image_processing, second one in bloblist_operations. The first phase may also be skipped if it was done before (as seen in output/bloblist)
3) Image_processing is used for extracting label from the image, making it easier to work with and extracting blob (glitter) coordinates from it, along with their other properties
4) Bloblist_operations is used for calculating hash, along with some experimental features

In output directory, it's possible to track the execution step by step. First phase is located in /output, second in output/report
/output/bloblist contains info of all blobs(glitter) of photo, /output/time contains time stats (may be useful to improve perfomance)

This project was created using JetBrains PyCharm, and it's probably recommended to use it

Contacts:
Aleksandr Bespalov: @Alex2184 (Telegram, preferred), sonkot25@mail.ru (email)
Mikhail Dvorkin: @mikhail_dvorkin (Telegram)

## Unorganized useful links

https://pypi.org/project/ImageHash/

https://github.com/JohannesBuchner/imagehash

https://pypi.org/project/dhash/

https://www.phash.org/docs/pubs/thesis_zauner.pdf

https://scikit-image.org/docs/stable/auto_examples/features_detection/	

http://ndl.ethernet.edu.et/bitstream/123456789/36780/1/15.pdf

https://dfrws.org/wp-content/uploads/2024/03/PHASER-Perceptual-hashing-algorithms-evaluati_2024_Forensic-Science-Interna.pdf

https://github.com/AabyWan/PHASER

https://www.researchgate.net/publication/377567576_Process_of_Fingerprint_Authentication_using_Cancelable_Biohashed_Template

https://link.springer.com/article/10.1007/s11042-015-2496-6

https://link.springer.com/article/10.1007/s11042-020-10135-w

https://link.springer.com/chapter/10.1007/978-981-10-2104-6_17

https://www.researchgate.net/publication/282970195_Perceptual_Hash_Function_based_on_Scale-Invariant_Feature_Transform_and_Singular_Value_Decomposition
