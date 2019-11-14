import cv2
import pandas as pd
import numpy as np
from torch import nn
import torch.utils.data
import matplotlib.pyplot as plt

dim_ = 4

class pc_regressor(nn.Module):
    def __init__(self):
        super(pc_regressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_, 1)
        )

    def forward(self, x):
        out = nn.Sigmoid()(self.fc(x) * 1.877537 + 4.92487) * 8.0 + 1
        return out

def cal_one(gray):
    gray = cv2.GaussianBlur(gray, (3, 5), cv2.BORDER_DEFAULT)
    mser = cv2.MSER_create(_delta=3, _min_area=1, _max_area=15)

    regions, boxes = mser.detectRegions(gray)
    remained_boxes = []

    for box in boxes:
        x, y, w, h = box
        ok = True
        for b in remained_boxes:
            x_, y_, w_, h_ = b
            if abs(x - x_ + (w - w_) / 2) + abs(y - y_ + (h - h_) / 2) < 10:
                ok = False
                break
        if ok:
            remained_boxes.append(box)

    return len(remained_boxes)

def particle_counter(img):
    shift = 35
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = np.array([
        cal_one(gray[shift:160, shift:265]),
        cal_one(gray[160:320, shift:265]),
        cal_one(gray[320:480, shift:265]),
        cal_one(gray[480:600, shift:265])
    ])
    print(w)
    return w

def get_sourses(path, file_list):
    img_list = []
    for f in file_list:
        img_list.append(cv2.imread(path + '/' + f + '.bmp'))
    return img_list


if __name__ == '__main__':
    torch.manual_seed(1111)

    df = pd.read_csv('train_scores.csv')
    img_list = get_sourses('train', df['image'])
    X = np.zeros((len(img_list), dim_))
    Y = df['total_rating'].to_numpy()

    for i in np.arange(len(img_list)):
        X[i] = particle_counter(img_list[i])

    df['r1'] = X[:, 0]
    df['r2'] = X[:, 1]
    df['r3'] = X[:, 2]
    df['r4'] = X[:, 3]

    plt.plot(X, Y, 'o', alpha=0.4)

    train_x = torch.from_numpy(X)
    train_y = torch.from_numpy(Y)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=1,     # mini batch size
        shuffle=True,
        num_workers=2,
    )

    loss_func = torch.nn.MSELoss()
    regressor = pc_regressor()
    opt = torch.optim.SGD(regressor.parameters(), lr=0.0001)

    for epoch in range(64):
        for step, (batch_x, batch_y) in enumerate(loader):
            pre = regressor(batch_x.float())
            loss = loss_func(pre, batch_y.float())

            print(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

    df_test = pd.read_csv('to_predict.csv')
    test_list = get_sourses('test', df_test['image'])
    X_test = np.ones((len(test_list), dim_))
    for i in np.arange(len(test_list)):
        X_test[i] = particle_counter(test_list[i])

    df_test['r1'] = X_test[:, 0]
    df_test['r2'] = X_test[:, 1]
    df_test['r3'] = X_test[:, 2]
    df_test['r4'] = X_test[:, 3]

    with torch.no_grad():
        pred = regressor(torch.from_numpy(X_test).float())
        out = torch.squeeze(pred)
        df_test['total_rating'] = out.numpy()

        pred = regressor(torch.from_numpy(X).float())
        out = torch.squeeze(pred)
        df['total_rating_pred'] = out.numpy()

    for p in regressor.parameters():
        print(p)
    plt.show()

    df.to_csv('trainS.csv', index=False)
    df_test.to_csv('testS.csv', index=False)