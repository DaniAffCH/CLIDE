from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ClassAdapter:
    _supportedClasses: List[str] = [
        "car",
        "person",
        "bike"
    ]
    _classesToId: Dict[str, int] = {c : idx for idx, c in enumerate(_supportedClasses)}

    @staticmethod
    def checkSubset(classes:List[str], mapping: Optional[Dict[str,str]] = None) -> bool:
        if mapping:
            classes, _ = ClassAdapter._adaptClasses(classes, mapping)

        return set(classes).issubset(set(ClassAdapter._supportedClasses))
    
    @staticmethod
    def _adaptClasses(classes:List[str], mapping: Dict[str,str]) -> Tuple[List[str], List[bool]]:
        mask = []
        adaptation = []

        for c in classes:
            if c in mapping and mapping[c] in ClassAdapter._supportedClasses:
                mask.append(True)
                adaptation.append(mapping[c])
            else:
                mask.append(False)


        return adaptation, mask
    
    @staticmethod
    def adaptClassesToName(classes:List[str], mapping: Dict[str,str]) -> Tuple[List[str], List[bool]]:
        classes, mask = ClassAdapter._adaptClasses(classes, mapping)

        if not ClassAdapter.checkSubset(classes):
            logger.error(f"The set of converted classes: {set(classes)} is not a subset of the supported classes: {set(ClassAdapter._supportedClasses)}")
            raise AssertionError("Some of the adapted classes are not supported.")

        return classes, mask
    
    @staticmethod
    def adaptClassesToId(classes:List[str], mapping: Dict[str,str]) -> Tuple[List[int], List[bool]]:
        classes, mask = ClassAdapter.adaptClassesToName(classes, mapping)

        return [ClassAdapter._classesToId[c] for c in classes], mask

