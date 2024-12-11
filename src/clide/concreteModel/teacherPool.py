import random
from clide.abstractModel.teacherModel import TeacherModel
from typing import Sequence, Optional, Tuple, Iterator, List

# Following singleton design pattern. There should be a unique TeacherPool
class TeacherPool:
    _instance = None  # Class-level attribute to hold the singleton instance

    def __new__(cls, modelsPool: Optional[Sequence[TeacherModel]] = None) -> 'TeacherPool':
        if cls._instance is None:
            if modelsPool is None:
                raise ValueError("Cannot create instance without a models pool.")
            cls._instance = super(TeacherPool, cls).__new__(cls)
            cls._instance._init_pool(modelsPool)
        return cls._instance

    def _init_pool(self, modelsPool: Sequence[TeacherModel]) -> None:
        self._pool = {model.name: model for model in modelsPool}
    
    def getModel(self, name: str) -> TeacherModel:
        return self._pool[name]
    
    def getRandomModel(self) -> TeacherModel:
        return random.choice(list(self._pool.values()))
    
    def teacherReviewersSplit(self) -> Tuple[TeacherModel,List[TeacherModel]]:
        if len(self._pool) < 2:
            raise ValueError("There must be at least two models in the pool to split.")

        teacher = random.choice(list(self._pool.values()))
        reviewers = [model for model in self._pool.values() if model != teacher]
        return teacher, reviewers
    
    def getAllModels(self) -> List[TeacherModel]:
        return list(self._pool.values())
    
    def __iter__(self) -> Iterator[TeacherModel]:
        return iter(self._pool.values())