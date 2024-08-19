from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import io
from PIL import Image
import matplotlib.pyplot as plt

def display_image(dbUri, dbName, image_id):
    client = MongoClient(dbUri)
    db = client[dbName]
    fs = gridfs.GridFS(db)

    # Retrieve image data from GridFS
    image_data = fs.get(ObjectId(image_id)).read()

    # Convert bytes data to an image and display it
    image = Image.open(io.BytesIO(image_data))
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == "__main__":
    dbUri = "mongodb://localhost:27017"
    dbName = "testing_db"
    image_id = "66b23684e8e7077270231c52"  # Replace with your image ObjectId as a string

    display_image(dbUri, dbName, image_id)
