import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.utils.data
import matplotlib.pyplot as plt

class pc_regressor(nn.Module):
    def __init__(self):
        super(pc_regressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x):
        return nn.Sigmoid()(self.fc(x) * 1.877537 + 4.92487) * 8.2 + 0.9

def particle_counter(img):
    shift = 35

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[shift:600, shift:300]

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


def get_sourses(path, file_list):
    img_list = []
    for f in file_list:
        img_list.append(cv2.imread(path + '/' + f + '.bmp'))
    return img_list


if __name__ == '__main__':
    torch.manual_seed(7)

    df = pd.read_csv('train_scores.csv')
    img_list = get_sourses('train', df['image'])
    X = np.zeros(len(img_list))
    Y = df['total_rating'].to_numpy()

    for i in np.arange(X.size):
        X[i] = particle_counter(img_list[i])

    plt.plot(X, Y, 'o', alpha=0.4)

    train_x = torch.from_numpy(X)
    train_x = torch.reshape(train_x, [X.size, 1])
    train_y = torch.from_numpy(Y)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=8,     # mini batch size
        shuffle=True,     # 要不要打乱数据 (打乱比较好)
        num_workers=2,    # 多线程来读数据
    )

    loss_func = torch.nn.MSELoss()
    regressor = pc_regressor()
    opt = torch.optim.SGD(regressor.parameters(), lr=0.0001)

    for epoch in range(32):  # 训练所有!整套!数据 3 次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            pre = regressor(batch_x.float())
            loss = loss_func(pre, batch_y.float())

            print(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

    df_test = pd.read_csv('to_predict.csv')
    test_list = get_sourses('test', df_test['image'])
    X_test = np.ones(len(test_list))
    for i in np.arange(X_test.size):
        X_test[i] = particle_counter(test_list[i])


    test_x = torch.reshape(torch.from_numpy(X_test), [X_test.size, 1])

    with torch.no_grad():
        pred = regressor(test_x.float())
        out = torch.squeeze(pred)
        df_test['total_rating'] = out.numpy()
        x_ = np.arange(70)
        out_ = regressor(torch.reshape(torch.from_numpy(x_), [x_.size, 1]).float())
        plt.plot(x_, out_.numpy())

    plt.legend(['groud truth', 'prediction'])
    plt.xlabel('Number of Particles')
    plt.ylabel('Total Rating')
    plt.title('Prediction of 1-D Sigmoid Regressor')
    plt.show()
    df_test.to_csv('team6submission.csv', index=False)