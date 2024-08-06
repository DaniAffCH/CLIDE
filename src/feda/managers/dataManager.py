from feda.abstractModel.teacherModel import TeacherModel
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from datetime import datetime
import io
from PIL import Image
import numpy as np

class DataManager:
    def __init__(self, dbUri: str, dbName: str, teacherModel: TeacherModel, maxSize: int) -> None:
        self._teacherModel = teacherModel
        self._maxSize = maxSize
        self._dbClient = MongoClient(dbUri)
        self._db = self._dbClient[dbName]
        self._fs = gridfs.GridFS(self._db)
        self._currentSize = self._calculateInitialSize()
        self.dbName = dbName

    def _calculateInitialSize(self) -> int:
        total_size = 0
        for file in self._db.fs.files.find():
            total_size += file['length']
        return total_size

    def _selectVictim(self):
        # TODO: find the correct way of selecting a victim. Currently FIFO.
        oldest = self._db.metadata.find_one(sort=[("upload_date", 1)])
        return oldest['_id'] if oldest else None

    def _annotateImage(self):
        # TODO: use a teacher to annotate 
        return 1.

    def addImage(self, imageData: np.ndarray):
        # Convert np.ndarray to bytes
        imagePil = Image.fromarray(imageData)
        with io.BytesIO() as output:
            imagePil.save(output, format="JPEG")
            imageBytes = output.getvalue()

        imageName = f"img_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        imageSize = len(imageBytes)

        # Ensure the dataset size does not exceed the maximum size
        while self._currentSize + imageSize > self._maxSize:
            victimId = self._selectVictim()
            if victimId:
                self.removeImage(victimId)
            else:
                raise RuntimeError("No image to remove but the dataset is over capacity.")


        self._currentSize += imageSize

        print(self._currentSize)

        # Store image using GridFS
        imageId = self._fs.put(imageBytes, filename=imageName)

        annotation = self._annotateImage()
        
        # Store metadata and annotation
        metadata = {
            '_id': imageId,
            'filename': imageName,
            'upload_date': datetime.now(),
            'annotation': annotation
        }
        self._db.metadata.insert_one(metadata)

    def getImage(self, imageId):
        imageData = self._fs.get(ObjectId(imageId)).read()

        metadata = self._db.metadata.find_one({'_id': ObjectId(imageId)})
        if metadata:
            return imageData, metadata['annotation']
        else:
            raise KeyError(f"No such image: {imageId}")
        
    def getDataset(self):
        # TODO
        pass

    def clean(self):
        # TODO
        pass

    def removeImage(self, image_id):
        imageLength = self._db.fs.files.find_one({'_id': ObjectId(image_id)})['length']
        self._fs.delete(ObjectId(image_id))
        self._db.metadata.delete_one({'_id': ObjectId(image_id)})
        self._currentSize -= imageLength 

    def __del__(self):
        self._dbClient.close()