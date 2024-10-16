from clide.abstractModel.studentModel import StudentModel
from clide.concreteModel.teacherPool import TeacherPool
from collections import OrderedDict
from dataclasses import dataclass
import torch.nn as nn
import torch
from typing import Dict, Any

@dataclass
class LayerInfo:
    modelName: str
    layerName: str
    outChannels: int
    isStudent: bool

@dataclass
class AdaptationLayer:
    studentLayerInfo: LayerInfo
    teacherLayerInfo: LayerInfo
    adaptationModule: nn.Module

class AdaptationKey:
    def __init__(self, studentName: str, teacherName: str, studentLayerName: str, teacherLayerName: str):
        self.studentName = studentName
        self.teacherName = teacherName
        self.studentLayerName = studentLayerName
        self.teacherLayerName = teacherLayerName

    def __hash__(self):
        return hash((self.studentName, self.teacherName, self.studentLayerName, self.teacherLayerName))

    def __eq__(self, other: Any) -> bool:
        return (self.studentName, self.teacherName, self.studentLayerName, self.teacherLayerName) == (other.studentName, other.teacherName, other.studentLayerName, other.teacherLayerName)

class FeatureDistillationManager:
    def __init__(self, studentModel: StudentModel, teacherPool: TeacherPool) -> None:
        self._adaptationMap: Dict[AdaptationKey, AdaptationLayer] = {}
        self._modelMap: Dict[str, StudentModel | TeacherPool] = {}

        studentLayers = studentModel.hookLayers
        studentModel.registerHooks(studentLayers)
        self._modelMap[studentModel.name] = studentModel

        for t in teacherPool:
            self._modelMap[t.name] = t
            teacherLayers = t.hookLayers
            
            if len(teacherLayers) != len(studentLayers):
                raise ValueError(
                    f"Layer count mismatch: Teacher model '{t.name}' has {len(teacherLayers)} layers, "
                    f"while student model '{studentModel.name}' has {len(studentLayers)} layers."
                )

            for sl, tl in zip(studentLayers, teacherLayers):
                soc = studentModel.getOutputChannels(sl)
                toc = t.getOutputChannels(tl)
                studentLayerInfo = LayerInfo(studentModel.name, sl, soc, True)
                teacherLayerInfo = LayerInfo(t.name, tl, toc, False)
                kernelSize = 3
                # TODO: fix data type, hardcoded for now
                adaptation = nn.Conv2d(soc, toc, kernelSize, padding=kernelSize // 2)

                key = AdaptationKey(studentModel.name, t.name, sl, tl)
                self._adaptationMap[key] = AdaptationLayer(studentLayerInfo, teacherLayerInfo, adaptation)

            t.registerHooks(teacherLayers)

    def getAdaptationLayer(self, studentName: str, teacherName: str, studentLayerName: str, teacherLayerName: str) -> AdaptationLayer:
        key = AdaptationKey(studentName, teacherName, studentLayerName, teacherLayerName)

        if key not in self._adaptationMap:
            raise KeyError(
                f"No layer connection found between student layer '{studentLayerName}' and teacher layer '{teacherLayerName}' "
                f"for the model pair '{studentName}' - '{teacherName}'. "
                "Ensure that the specified layers have been correctly registered for distillation."
            )
        
        return self._adaptationMap[key]

    def getModel(self, modelName: str) -> Any:
        if modelName not in self._modelMap:
            raise KeyError(f"Model '{modelName}' is not registered in the manager.")
        return self._modelMap[modelName]
    
    def updateModel(self, model: StudentModel):
        if model.name not in self._modelMap:
            raise KeyError(f"Model {model.name} is not a registered model")
        modelLayers = model.hookLayers
        model.registerHooks(modelLayers)
        self._modelMap[model.name] = model
    
    def getFeatures(self, modelName: str) -> Dict[str, torch.Tensor]:
        model = self.getModel(modelName)
        return model.getHooks()
    
    def getAdaptedFeatures(self, studentModelName: str, teacherModelName: str) -> Dict[str, OrderedDict]:
        studentFeatures = self.getFeatures(studentModelName)
        teacherFeatures = self.getFeatures(teacherModelName)

        adaptedFeatures = {
            "teacher": OrderedDict(),
            "student": OrderedDict()
        }
        
        for (studentFeatureName, studentFeature), (teacherFeatureName, teacherFeature) in zip(studentFeatures.items(), teacherFeatures.items()):
            adaptation = self.getAdaptationLayer(studentModelName, teacherModelName, studentFeatureName, teacherFeatureName)

            # TODO: studentModel.requires_grad FIX THIS
            adaptation.adaptationModule.requires_grad_(True)

            adaptation.adaptationModule.to(studentFeature.device)

            adaptedFeatures["teacher"][teacherFeatureName] = teacherFeature
            adaptedFeatures["student"][studentFeatureName] = adaptation.adaptationModule(studentFeature)

            assert (
                adaptedFeatures["teacher"][teacherFeatureName].shape == adaptedFeatures["student"][studentFeatureName].shape
            ), (
                f"Adapted features don't have the same shape:\n"
                f"Teacher {teacherModelName} feature '{teacherFeatureName}' shape: {adaptedFeatures['teacher'][teacherFeatureName].shape},\n"
                f"Student {studentModelName} feature '{studentFeatureName}' shape: {adaptedFeatures['student'][studentFeatureName].shape}."
            )
        return adaptedFeatures
