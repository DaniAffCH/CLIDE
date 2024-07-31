from feda.concreteModels.florence2 import Florence2, FlorenceModelType

import requests
from PIL import Image

if __name__ == "__main__":
    m = Florence2(FlorenceModelType.FLORENCE_2_BASE)
    m.prompt = "<OD>"
    url = "https://cdn.apartmenttherapy.info/image/upload/f_jpg,q_auto:eco,c_fill,g_auto,w_1500,ar_4:3/k%2Farchive%2F7c7ef71dc7e19f1e9262bb724203a1014dc230e8"
    image = Image.open(requests.get(url, stream=True).raw)

    m.forward(image)