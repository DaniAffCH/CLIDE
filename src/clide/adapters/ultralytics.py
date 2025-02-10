import torch

@torch.no_grad()
def batchedTeacherPredictions(teacher, images):        
    gt = teacher.model(images)

    all_bboxes, all_batch_idx, all_cls, all_conf = [], [], [], []

    for i, el in enumerate(gt):
        if el.boxes is not None and el.boxes.xywhn.numel() > 0:
            all_bboxes.append(el.boxes.xywhn.clone().detach())
            all_batch_idx.append(torch.full((el.boxes.xywhn.shape[0],), i, device=el.boxes.xywhn.device))
            all_cls.append(el.boxes.cls.clone().detach())
            all_conf.append(el.boxes.conf.clone().detach())

    return {
        "batch_idx": torch.cat(all_batch_idx) if all_batch_idx else torch.empty(0, device=images.device),
        "cls": torch.cat(all_cls) if all_cls else torch.empty((0,1), device=images.device),
        "bboxes": torch.cat(all_bboxes) if all_bboxes else torch.empty((0, 4), device=images.device),
        "conf": torch.cat(all_conf) if all_conf else torch.empty(0, device=images.device),
    }