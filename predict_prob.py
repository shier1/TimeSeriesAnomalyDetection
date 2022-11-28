import platform
import pandas as pd
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from glob import glob
from models.modeling import get_model
from utils.data_utils import load_data
from torch.utils.data import Dataset, DataLoader

def predict(model, val_loader, device):
    print("***** Running Predict *****")

    model.eval()
    all_preds = []
    epoch_iterator = tqdm(val_loader,
                          desc="Predicting... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

    for step, x in enumerate(epoch_iterator):
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            preds = F.softmax(logits, -1)
            #preds = logits

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Predicting...")
    all_preds = all_preds[0]
    return all_preds


class TestDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data_list = data_list

    def __len__(self,):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data_list[idx, :]).float()

def save_res(test_pkl_files, scores):
    #记录文件名和对应的异常得分
    predict_result={}
    for i in tqdm(range(len(test_pkl_files))):
        file=test_pkl_files[i]
        #如果是window系统：
        if platform.system().lower() == 'windows':
            name=file.split('\\')[-1]
        #如果是linux系统
        elif platform.system().lower() == 'linux':
            name=file.split('/')[-1]
        predict_result[name]=scores[i]

    predict_score=pd.DataFrame(list(predict_result.items()),columns=['file_name','score'])#列名必须为这俩个
    predict_score.to_csv('submision.csv',index = False)

if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = get_model()
    model_state_dict = torch.load('./output/Conv-Fc-8-4-2-2-0_checkpoint.bin')
    # model_state_dict = torch.load('output/last.bin')
    model.load_state_dict(model_state_dict)
    model.to(device)

    train_data_path = './Train'
    train_pkl_files = glob(train_data_path+"/*.pkl")
    random.shuffle(train_pkl_files)
    train_data, train_label, train_mileage = load_data(train_pkl_files, label=True)
    train_data_max = np.max(train_data)
    train_data_min = np.min(train_data)
    mileage_min = np.min(train_mileage)
    mileage_max = np.max(train_mileage)

    test_data_path = './Test_A'
    test_pkl_files = glob(test_data_path+"/*.pkl")
    test_data, _, test_mileages = load_data(test_pkl_files, label=False)

    test_data = (test_data-train_data_min) / (train_data_max-train_data_min)
    test_mileages = (test_mileages - mileage_min) / (mileage_max - mileage_min)
    test_mileages = np.expand_dims(test_mileages, axis=2)
    val_mileages = test_mileages.repeat(256, axis=1)
    test_data = np.concatenate([test_data, val_mileages], axis=2)

    test_dataset = TestDataset(test_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
    
    all_preds = predict(model, test_dataloader, device)

    print(all_preds)
    scores = all_preds[:, 1]
    print(scores)
    save_res(test_pkl_files, scores)
