import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维数字最大就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层，输出维度为类别数
        self.activation = torch.softmax  # softmax归一化函数，将输出转换为概率分布
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, class_num)
        y_pred = self.activation(x, dim=1)  # (batch_size, class_num) -> (batch_size, class_num)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失（注意：交叉熵损失直接使用线性层输出，不需要softmax）
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，哪一维数字最大就属于第几类
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 找到最大值所在的索引作为标签
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类别样本数量
    class_count = [0] * 5
    for label in y:
        class_count[label] += 1
    print(f"本次预测集中各类别样本数量：{class_count}")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            predicted_class = torch.argmax(y_p).item()  # 找到概率最大的类别
            if predicted_class == y_t.item():
                correct += 1  # 预测正确
            else:
                wrong += 1
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong):.4f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别数
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()  # 找到概率最大的类别
        max_prob = torch.max(res).item()  # 最大概率值
        print(f"输入：{vec}, 预测类别：{predicted_class}, 最大概率值：{max_prob:.4f}")  # 打印结果


if __name__ == "__main__":
    main()
    # 测试向量，每个向量的最大值所在位置分别为第4、2、0、0维
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 第4维最大
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],  # 第2维最大
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 第0维最大
                [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]  # 第0维最大
    predict("model.bin", test_vec)
