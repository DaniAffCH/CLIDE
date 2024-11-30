import torch

@torch.no_grad()
def batchedTeacherPredictions(teacher, images):        
    gt = teacher.model(images)

    all_bboxes = []
    all_batch_idx = []
    all_cls = []
    all_conf = []

    for i, el in enumerate(gt):
        bbox = el.boxes.xywhn.clone().detach()
        batch_idx = torch.full((bbox.shape[0],), i, device=bbox.device)
        cls = el.boxes.cls.clone().detach()
        conf = el.boxes.conf.clone().detach() 

        all_bboxes.append(bbox)
        all_batch_idx.append(batch_idx)
        all_cls.append(cls)
        all_conf.append(conf)

    concatenated_bboxes = torch.cat(all_bboxes, dim=0)
    concatenated_batch_idx = torch.cat(all_batch_idx, dim=0)
    concatenated_cls = torch.cat(all_cls, dim=0)
    concatenated_conf = torch.cat(all_conf, dim=0)    

    return {
        "batch_idx": concatenated_batch_idx,
        "cls": concatenated_cls,
        "bboxes": concatenated_bboxes,
        "conf": concatenated_conf,  # Include confidences in the output
    }