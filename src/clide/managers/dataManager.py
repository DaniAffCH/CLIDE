from dataclasses import dataclass
from clide.concreteModel.teacherPool import TeacherPool
from clide.distiller.importanceDistiller import ImportanceDistiller
from pymongo import MongoClient
from typing import Dict
import gridfs
from bson.objectid import ObjectId
from datetime import datetime
from typing import List
import humanfriendly
import io
import numpy as np
import logging
import torch
import time
import wandb
import random


logger = logging.getLogger(__name__)

@dataclass
class Sample:
    image: torch.Tensor
    importanceMap: Dict[str, torch.Tensor]
    annotation: int 

class DataManager:
    def __init__(self, dbUri: str, dbName: str, teacherPool: TeacherPool, maxSize: str, updateRate: int, unusedRatioThreshold: float, minCollectingTime: int, useImportanceEstimation: bool, _callbacks, cleanFirst: bool = False) -> None:
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
        self._useImportanceEstimation = useImportanceEstimation
        self._callbacks = _callbacks
        assert 0. <= unusedRatioThreshold <= 1., "unusedRatioThreshold must be between 0 and 1 (included)"
        self._minCollectingTime = minCollectingTime

        # |-_|
        if self._useImportanceEstimation:
            self.importanceDistiller = ImportanceDistiller([(1,1),(2,2),(4,4),(8,8),(16,16)], [0.5,0.6,0.6,0.7,0.7], 640, 64)

        logger.info(f"Connection to db established")
        self.checkAndRestoreConsistency()
        
    def getNumSamples(self) -> int:
        return self._db.metadata.count_documents({})
        
        
    def run_callbacks(self, callback:str):
        # TODO: should be a parameter
        if time.time() - self._loggingTimer > 15 and wandb:
            self._loggingTimer = time.time()
            for callbackHandler in self._callbacks[callback]:
                callbackHandler(self)
        
    def startCollectionSession(self):
        self._collectionTimer = time.time()

    def _calculateInitialSize(self) -> int:
        total_size = 0
        for file in self._db.fs.files.find():
            total_size += file['length']
        return total_size

    def _selectVictim(self):
        # TODO: find the correct way of selecting a victim. Currently RANDOM.

        random_doc = list(self._db.metadata.aggregate([{"$sample": {"size": 1}}]))
        return random_doc[0] if random_doc else None

    def _annotateImage(self, imageData: np.ndarray) -> Dict[str, torch.Tensor]:
        annotator = self._teacherPool.getRandomModel()
        label = annotator.forward(torch.tensor(imageData))

        # TODO: this should be related to the type of task. Assuming Detection for now
        return label["output"].classes
    
    def _estimateFeatureImportance(self, imageData: np.ndarray):
        return {t.name: self.importanceDistiller(t,imageData).cpu() for t in self._teacherPool}

    def _unusedRatio(self) -> float:
        '''
        Returns the ratio of images where "used" is False.
        '''
        total_images = self._db.metadata.count_documents({})
        unused_images = self._db.metadata.count_documents({"used": False})
        
        if total_images == 0:
            return 0.0  
        
        ratio = unused_images / total_images

        return ratio

    def checkAndRestoreConsistency(self):
        # Get all _id values from the metadata collection
        metadata_cursor = self._db.metadata.find({}, {'_id': 1, 'importance_ids': 1})
        
        # Collect all file IDs referenced by metadata
        used_files = set()
        
        # Collect referenced files (image _id + importance_id list)
        for metadata in metadata_cursor:
            # Add the image _id to used_files
            image_id = metadata['_id']
            used_files.add(str(image_id))  # Add image _id to used_files
            
            # Add all importance_id values to used_files
            if "importance_ids" in metadata:
                for importance_id in metadata['importance_ids'].values():
                    used_files.add(str(importance_id))  # Add importance_id to used_files

        # Get all file IDs in fs.files
        fs_files = self._db.fs.files.find()
        fs_file_ids = {str(file['_id']): file for file in fs_files}  # Map of file _id to file document in fs

        # Find unreferenced files (i.e., files that are in fs but not in used_files)
        pending_files = set(fs_file_ids.keys()) - used_files
        
        # Handle unreferenced files
        if pending_files:
            for file_id in pending_files:
                file_doc = fs_file_ids[file_id]
                file_name = file_doc.get('filename', 'unknown')
                
                try:
                    self._deleteFile(file_id) 
                    logger.info(f"Removed unreferenced file: {file_name} ({file_id})")
                except Exception as e:
                    logger.error(f"Error removing file {file_id}: {str(e)}")
        else:
            logger.info("No unreferenced files to remove in fs.")
        
        logger.info("Consistency check and restoration completed.")

    def addImage(self, imageData: np.ndarray):
            # limit the update frequency
            if time.time() - self._updateTimer < self._updateRate:
                return
            with self._dbClient.start_session() as session:
                with session.start_transaction():
                    # Convert np.ndarray to bytes
                    image = torch.tensor(imageData)

                    annotation = self._annotateImage(imageData)
                    
                    with io.BytesIO() as output:
                        torch.save(image, output)
                        imageSerialized = output.getvalue()
                        
                    if self._useImportanceEstimation:
                        importanceMap = self._estimateFeatureImportance(imageData)
                        with io.BytesIO() as buffer:
                            importanceMapSerialized = dict()
                            
                            for n, i in importanceMap.items():
                                buffer.seek(0)
                                buffer.truncate(0)
                                torch.save(i, buffer)
                                importanceMapSerialized[n] = buffer.getvalue()

                    now = datetime.now().strftime('%Y%m%d%H%M%S%f')

                    imageSize = len(imageSerialized)
                    requestSize = imageSize
                    
                    if self._useImportanceEstimation:
                        importanceSize = sum(len(v) for v in importanceMapSerialized.values())      
                        requestSize += importanceSize   
                        
                    imageName = f"img_{now}.jpg"

                    # Ensure the dataset size does not exceed the maximum size
                    while self._currentSize + requestSize > self._maxSize:
                        victim = self._selectVictim()
                        if victim:
                            self.removeSample(victim)
                        else:
                            raise RuntimeError("No image to remove but the dataset is over capacity.")

                    self._updateTimer = time.time()

                    self._currentSize += requestSize

                    imageId = self._fs.put(imageSerialized, filename=imageName)
                    
                    # Store metadata and annotation
                    metadata = {
                        '_id': imageId,
                        'filename': imageName,
                        'upload_date': datetime.now(),
                        'annotation': annotation,
                        'used': False,
                        'usedForTraining': False
                    }
                    
                    if self._useImportanceEstimation:
                        importanceIds = {k : self._fs.put(v, filename=f"imp_{now}_{k}.pt") for k,v in importanceMapSerialized.items()}
                        metadata['importance_ids'] = importanceIds
                        
                    self._db.metadata.insert_one(metadata)
                    
            self.run_callbacks("on_data_update")

    def getSample(self, imageId) -> Sample:
        imageData = self._fs.get(ObjectId(imageId)).read()

        with io.BytesIO(imageData) as input_buffer:
            image = torch.load(input_buffer)

        metadata = self._db.metadata.find_one({'_id': ObjectId(imageId)})

        importanceMap = {}
        
        if self._useImportanceEstimation:
            for k, importanceId in metadata["importance_ids"].items():
                importanceData = self._fs.get(ObjectId(importanceId)).read()
                
                with io.BytesIO(importanceData) as input_buffer:
                    importance = torch.load(input_buffer)

                importanceMap[k] = importance

        return Sample(image, importanceMap, metadata["annotation"])

    def getAllIds(self) -> List[str]:
        '''
        Returns a List of sample ids.
        '''
        dataset = []
        cursor = self._db.metadata.find({}, {'_id': 1})
        for doc in cursor:
            dataset.append(str(doc['_id']))

        self._db.metadata.update_many({}, {"$set": {"used": True}})

        return dataset
    
    def getForValidation(self, n: int) -> List[str]:
        '''
        Returns a List of n sample ids where the field "usedForTraining" is False.
        '''
        # Fetch samples that have "usedForTraining" as False
        cursor = self._db.metadata.find({"usedForTraining": False}, {'_id': 1})
        all_ids = [doc['_id'] for doc in cursor]
        
        # If there are less samples than requested, return all
        val_ids = random.sample(all_ids, min(n, len(all_ids)))
        
        self._db.metadata.update_many(
            {"_id": {"$in": val_ids}},
            {"$set": {"used": True}}
        )
    
        return list(map(str,val_ids))

    def getForTraining(self, n: int, validationSet: List[str] = []) -> List[str]:
        '''
        Returns a List of n sample ids where the field "usedForTraining" is False,
        and updates their "usedForTraining" field to True.
        '''
        cursor = self._db.metadata.find({}, {'_id': 1})
        # Make sure that train and val remains disjointed
        all_ids = [doc['_id'] for doc in cursor if str(doc["_id"]) not in validationSet]
        
        # Select random samples
        selected_ids = random.sample(all_ids, min(n, len(all_ids)))
        
        # Update the selected samples' "usedForTraining" field to True
        self._db.metadata.update_many(
            {"_id": {"$in": selected_ids}},
            {"$set": {"usedForTraining": True}}
        )

        self._db.metadata.update_many(
            {"_id": {"$in": selected_ids}},
            {"$set": {"used": True}}
        )
        
        return list(map(str, selected_ids))

    def stopCollecting(self) -> bool:
        minTimeElapsed = time.time() - self._collectionTimer > self._minCollectingTime
        unused = self._unusedRatio()
        return unused > self._unusedRatioThreshold and minTimeElapsed

    def clean(self):
        logger.info(f"Cleaning database {self.dbName}")
        self._dbClient.drop_database(self.dbName)

    def _deleteFile(self, id: str):
        self._fs.delete(ObjectId(id))

    def _deleteMetadata(self, id: str):
        self._db.metadata.delete_one({'_id': ObjectId(id)})

    def removeSample(self, metadata):
        with self._dbClient.start_session() as session:
            with session.start_transaction():

                image_id = metadata['_id']

                importance_ids = metadata.get('importance_ids', {}).values() if 'importance_ids' in metadata else []

                imageLength = self._db.fs.files.find_one({'_id': ObjectId(image_id)})['length']

                self._deleteFile(image_id)
                self._deleteMetadata(image_id)

                self._currentSize -= imageLength

                for importance_id in importance_ids:
                    importanceLength = self._db.fs.files.find_one({'_id': ObjectId(importance_id)})['length']
                    self._deleteFile(importance_id)
                    self._currentSize -= importanceLength

    def __del__(self):
        self._dbClient.close()