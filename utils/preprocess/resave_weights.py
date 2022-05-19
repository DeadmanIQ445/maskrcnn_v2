import torch

def clean_model_for_resume(pth, output_pth):
    m = torch.load(str(pth))
    to_remove = ['scheduler', 'iteration']
    for i in to_remove:
        if i in m:
            del m[i]
    torch.save(m, output_pth)

clean_model_for_resume('../../weights/aerial_summer_pieceofland.pth', '../../weights/aerial_summer_pieceofland_cleaned.pth')