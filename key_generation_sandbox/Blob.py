
import json
class Blob:
    def __init__(self, coords, size, brightness=None, distinctiveness=None, dct_128_8=None, bmp_128_7 = None, bmp_128_15 = None, color=None):
        self.coords = coords
        self.size = size
        self.brightness = brightness
        self.distinctiveness = distinctiveness
        self.dct_128_8 = dct_128_8
        self.bmp_128_7 = bmp_128_7
        self.bmp_128_15 = bmp_128_15
    def to_json(self):
        return json.dumps({
            "coords": self.coords,
            "size": self.size,
            "brightness": self.brightness,
            "distinctiveness": self.distinctiveness,
            "dct_128_8": self.dct_128_8,
            "bmp_128_7": self.bmp_128_7,
            "bmp_128_15": self.bmp_128_15,
        })
    def log(self, file):
        file.write(self.to_json() + "\n")
    def same_dot(self, blob2):
        return self.coords[0] == blob2.coords[0] and self.coords[1] == blob2.coords[1]
    @staticmethod
    def unpack(blob_json):
        blob_dict = json.loads(blob_json)
        coords = blob_dict["coords"]
        size = blob_dict["size"]
        brightness = blob_dict["brightness"]
        distinctiveness = blob_dict["distinctiveness"]
        dct_128_8 = blob_dict["dct_128_8"]
        bmp_128_7 = blob_dict["bmp_128_7"]
        bmp_128_15 = blob_dict["bmp_128_15"]
        return Blob(coords, size, brightness, distinctiveness, dct_128_8, bmp_128_7, bmp_128_15)