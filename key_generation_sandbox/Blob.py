
import json
class Blob:
    def __init__(self, coords, size, brightness=None, dct_128_8=None, color=None):
        self.coords = coords
        self.size = size
        self.brightness = brightness
        self.dct_128_8 = dct_128_8
    def to_json(self):
        return json.dumps({
            "coords": self.coords,
            "size": self.size,
            "brightness": self.brightness,
            "dct_128_8": self.dct_128_8,
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
        dct_128_8 = blob_dict["dct_128_8"]
        return Blob(coords, size, brightness, dct_128_8)