from feda.abstractModel.studentModel import StudentModel
from feda.concreteModel.teacherPool import TeacherPool
from feda.managers.dataManager import DataManager


class DistillationTrainer:
    def __init__(self, studentModel: StudentModel, teacherPool: TeacherPool, dataManager: DataManager) -> None:
        pass