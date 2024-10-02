from feda.concreteModel.teacherPool import TeacherPool
from pymongo import MongoClient
from typing import Dict
import gridfs
from bson.objectid import ObjectId
from datetime import datetime
from typing import List
import humanfriendly
import io
from PIL import Image
import numpy as np
import logging
import torch
import time

logger = logging.getLogger(__name__)
class DataManager:
    def __init__(self, dbUri: str, dbName: str, teacherPool: TeacherPool, maxSize: str, updateRate: int, unusedRatioThreshold: float, minCollectingTime: int, cleanFirst: bool = False) -> None:
        logger.info(f"Connecting to dbUri: {dbUri} dbName: {dbName}")
        self.dbName = dbName
        self._teacherPool = teacherPool
        self._maxSize = humanfriendly.parse_size(maxSize)
        self._dbClient = MongoClient(dbUri)
        if cleanFirst:
            self.clean()
        self._db = self._dbClient[dbName]
        self._fs = gridfs.GridFS(self._db)
        self._currentSize = self._calculateInitialSize()
        self._collectionTimer = time.time()
        self._updateTimer = time.time()
        self._loggingTimer = time.time()
        self._updateRate = updateRate # in seconds 
        self._unusedRatioThreshold = unusedRatioThreshold
        assert 0. <= unusedRatioThreshold <= 1., "unusedRatioThreshold must be between 0 and 1 (included)"
        self._minCollectingTime = minCollectingTime

        logger.info(f"Connection to db established")
        self._restoreConsistency()

    def startCollectionSession(self):
        self._collectionTimer = time.time()

    def _restoreConsistency(self):
        pipeline = [
        {
            "$lookup": {
                "from": "metadata",
                "localField": "_id",
                "foreignField": "_id",
                "as": "match"
            }
        },
        {
            "$unwind": {
                "path": "$match",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$match": {
                "match": None
            }
        },
        {
            "$project": {
                "_id": 1
            }
        }
        ]

        missingKeys = self._db.fs.files.aggregate(pipeline)

        changed = False
        for res in missingKeys:
            self._deleteFile(res["_id"])
            changed = True
        
        if changed:
            logger.info(f"Database consistency restored.")

        # TODO: is the other way around needed?


    def _calculateInitialSize(self) -> int:
        total_size = 0
        for file in self._db.fs.files.find():
            total_size += file['length']
        return total_size

    def _selectVictim(self):
        # TODO: find the correct way of selecting a victim. Currently RANDOM.

        random_doc = list(self._db.metadata.aggregate([{"$sample": {"size": 1}}]))
        return random_doc[0]['_id'] if random_doc else None

    def _annotateImage(self, imageData: np.ndarray) -> Dict[str, torch.Tensor]:
        annotator = self._teacherPool.getRandomModel()
        label = annotator.forward(torch.tensor(imageData))

        # TODO: this should be related to the type of task. Assuming Detection for now
        return label["output"].classes

    def _unusedRatio(self) -> float:
        '''
        Returns the ratio of images where "used" is False.
        '''
        total_images = self._db.metadata.count_documents({})
        unused_images = self._db.metadata.count_documents({"used": False})

        if total_images == 0:
            return 0.0  
        
        ratio = unused_images / total_images

        if time.time() - self._loggingTimer > 16:
            logger.info(f"New samples ratio {ratio:.3f} (threshold {self._unusedRatioThreshold})")
            self._loggingTimer = time.time()

        return ratio

    def _checkConsistency(self):
        fsFilesN = self._db.fs.files.count_documents({})
        metadataN = self._db.metadata.count_documents({})

        if fsFilesN != metadataN:
            error_message = f"Filesystem - metadata mismatch: fs.files count = {fsFilesN}, metadata count = {metadataN}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    def addImage(self, imageData: np.ndarray):
            # limit the update frequency
            if time.time() - self._updateTimer < self._updateRate:
                return
            with self._dbClient.start_session() as session:
                with session.start_transaction():
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

                    self._updateTimer = time.time()

                    self._currentSize += imageSize

                    imageId = self._fs.put(imageBytes, filename=imageName)

                    annotation = self._annotateImage(imageData)

                    # Store metadata and annotation
                    metadata = {
                        '_id': imageId,
                        'filename': imageName,
                        'upload_date': datetime.now(),
                        'annotation': annotation,
                        'used': False
                    }
                    self._db.metadata.insert_one(metadata)

                    self._checkConsistency()

    def getImage(self, imageId):
        imageData = self._fs.get(ObjectId(imageId)).read()

        metadata = self._db.metadata.find_one({'_id': ObjectId(imageId)})
        if metadata:
            return imageData, metadata['annotation']
        else:
            raise KeyError(f"No such image: {imageId}")
        
    def getAllIds(self) -> List[str]:
        '''
        Returns a List of image ids.
        '''
        dataset = []
        cursor = self._db.metadata.find({}, {'_id': 1})
        for doc in cursor:
            dataset.append(str(doc['_id']))

        # Update all images' "used" field to True
        self._db.metadata.update_many({}, {"$set": {"used": True}})

        return dataset

    def stopCollecting(self) -> bool:
        minTimeElapsed = time.time() - self._collectionTimer > self._minCollectingTime
        unused = self._unusedRatio()
        return True
        return unused > self._unusedRatioThreshold and minTimeElapsed

    def clean(self):
        logger.info(f"Cleaning database {self.dbName}")
        self._dbClient.drop_database(self.dbName)

    def _deleteFile(self, id: str):
        self._fs.delete(ObjectId(id))

    def _deleteMetadata(self, id: str):
        self._db.metadata.delete_one({'_id': ObjectId(id)})

    def removeImage(self, image_id):
        with self._dbClient.start_session() as session:
            with session.start_transaction():
                imageLength = self._db.fs.files.find_one({'_id': ObjectId(image_id)})['length']
                
                self._deleteFile(image_id)
                self._deleteMetadata(image_id)
                
                self._currentSize -= imageLength

    def __del__(self):
        self._dbClient.close()