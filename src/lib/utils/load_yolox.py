from lib.model.yolo.yolox.exp.build import get_exp
import torch
import os


def exetract_feature(img):
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
#     model_ckpt='/data/MOT/Modified/AIMAX/c2/src/lib/model/yolo/best_ckpt.pth'
#     root=os.getcwd()
    exp_file=os.path.join(os.getcwd(),'lib/model/yolo/exps/example/yolox_voc/yolox_voc_s.py')
    model_name='yolox-s'
    model_ckpt=os.path.join(os.path.pardir,'pretrained/yolox.pth')
    exp = get_exp(exp_file, model_name)

    model = exp.get_model()

    #model.to(device)
    model.cuda()
    model.eval()


    ckpt_file = model_ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    with torch.no_grad():
            features,_ = model(img)
            
    return features
