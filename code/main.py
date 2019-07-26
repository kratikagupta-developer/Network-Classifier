import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import Net
import torch.nn as nn
import sklearn.exceptions
from sklearn.metrics import f1_score, recall_score, precision_score
import os.path
import dataprocessor
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

pd.set_option('display.max_rows', 75)

# Source: https://www.youtube.com/watch?v=zN49HdDxHi8&t=0s&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&index=9


class TrafficDataset(Dataset):
    """ Anonymous Traffic dataset """

    # Initialize csv data
    def __init__(self, is_train_set=False):
        # Fields to use when reading in pandas subject to change
        fields = ['duration', 'layer4_proto', 'numPktsSnt',
                  'numPktsRcvd', 'numBytesSnt', 'numBytesRcvd', 'minPktSz', 'maxPktSz', 'avePktSize', 'pktps', 'bytps',
                  'pktAsm', 'bytAsm', 'ip_mindIPID', 'ip_maxdIPID', 'ip_minTTL', 'ip_maxTTL', 'ip_TTL_Chg', 'ip_TOS',
                  'ip_flags', 'tcp_PSeqCnt',
                  'tcp_SeqSntBytes', 'tcp_SeqFaultCnt', 'tcp_PAckCnt', 'tcp_FlwLssAckRcvdBytes', 'tcp_AckFaultCnt',
                  'tcp_InitWinSz', 'tcp_AveWinSz', 'tcp_MinWinSz', 'tcp_MaxWinSz', 'tcp_WinSzDwnCnt', 'tcp_WinSzUpCnt',
                  'tcp_WinSzChgDirCnt', 'tcp_AggrFlags', 'tcp_AggrAnomaly',
                  'tcp_MSS', 'tcp_WS', 'tcp_OptCnt', 'tcp_S-SA/SA-A_Trip', 'tcp_S-SA-A/A-A_RTT',
                  'tcp_RTTAckTripMin', 'tcp_RTTAckTripMax', 'tcp_RTTAckTripAve', 'tcpStates', 'connSrc', 'connDst',
                  'connSrc<->Dst', 'min_pl', 'max_pl', 'mean_pl', 'low_quartile_pl', 'median_pl', 'upp_quartile_pl',
                  'iqd_pl', 'mode_pl', 'range_pl', 'std_pl', 'stdrob_pl', 'skew_pl', 'exc_pl', 'min_iat', 'max_iat',
                  'mean_iat',  'low_quartile_iat', 'median_iat', 'upp_quartile_iat', 'iqd_iat', 'mode_iat', 'range_iat',
                  'std_iat', 'stdrob_iat', 'skew_iat', 'exc_iat', 'anon_name']

        if is_train_set:
            self.data = pd.read_csv('./data/Anon17csv/train_data.csv', header=0, usecols=fields)
            self.data.dropna(inplace=True)
        else:
            self.data = pd.read_csv('./data/Anon17csv/test_data.csv', header=0, usecols=fields)
            self.data.dropna(inplace=True)

        # Convert Hexadecimal fields to decimal
        self.data['ip_TOS'] = self.data.ip_TOS.apply(lambda x: int(x, 16))
        self.data['ip_flags'] = self.data.ip_flags.apply(lambda x: int(x, 16))
        self.data['tcp_AggrAnomaly'] = self.data.tcp_AggrAnomaly.apply(lambda x: int(x, 16))
        self.data['tcpStates'] = self.data.tcpStates.apply(lambda x: int(x, 16))
        self.data['tcp_AggrFlags'] = self.data.tcpStates.astype('str').apply(lambda x: int(x, 16))

        # Shuffle the data set
        self.data = shuffle(self.data)
        # Number of columns in data is 75
        # Number of columns for self.x or input is 73
        # Nunber of columns for self.y or target is 1
        self.x = torch.from_numpy(np.asarray(self.data.iloc[:, :-1])).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.asarray(self.data.iloc[:, -1])).type(torch.LongTensor)

        self.len = int(len(self.data))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def train():
    total_loss = 0
    for i, data in enumerate(train_loader, 1):
        # get the inputs and labels from data loader
        inputs, labels = data
        ypred_var = net(inputs)
        loss = criterion(ypred_var, labels)
        total_loss += loss.item()

        net.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                epoch,  i * len(inputs), len(train_loader.dataset),
                100. * i * len(inputs) / len(train_loader.dataset),
                total_loss / i * len(inputs)))
        plot_loss.append(total_loss)


def test():
    test_set = TrafficDataset()
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    print("evaluating trained model ...")
    '''
    correct = 0
    train_data_size = len(test_loader.dataset)

    for inputs, labels in test_loader:
        output = net(inputs)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))
    '''

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test set Accuracy: {:.2f}% '.format(100 * correct / total))
        plot_accuracy.append(correct / total)
        print('precision: ', precision_score(y_true=labels, y_pred=predicted, average="weighted"))
        print('recall: ', recall_score(y_true=labels, y_pred=predicted, average="weighted"))
        print('f1-score: ', f1_score(y_true=labels, y_pred=predicted, average="weighted"))
        print('\n' + classification_report(labels, predicted))


if __name__ == '__main__':
    if not os.path.exists("./data/Anon17csv/test_data.csv") or not os.path.exists("./data/Anon17csv/train_data.csv"):
        print("Getting Data")
        dataprocessor.prepare_data()

    # Hyper parameters
    num_epochs = 25
    learning_rate = 0.0001
    batch_size = 128

    # This setting of hyper parameters gives the following results:
    # Test set Accuracy: 99.59%
    # precision :  0.9792
    # recall:  0.9895
    # f1-score:  0.9843
    
    # Call Data processor here to invoke train-test split ONCE
    train_set = TrafficDataset(is_train_set=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    # Initialize model
    net = Net()

    # Initialize loss
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # List for loss values
    plot_loss = []

    # List for accuracy values
    plot_accuracy = []
    
    print("Training for %d epochs..." % num_epochs)
    for epoch in range(1, num_epochs + 1):
        train()
        test()

    # Plot loss values
    plt.ylabel('Loss')
    plt.xlabel("Training Steps")
    plt.title('Loss over progression of training steps')
    plt.plot([i for i in range(len(plot_loss))], sorted(plot_loss, reverse=True))
    plt.show()

    # Plot accuracy values
    plt.ylabel('Accuracy')
    plt.xlabel("Training epoch")
    plt.title('Accuracy over progression of training epoch')
    plt.plot([i for i in range(len(plot_accuracy))], plot_accuracy)
    plt.show()

    # Correlation Matrix
    df = pd.read_csv("./data/Anon17csv/train_data.csv", header=0, index_col=0)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    ax.set_title('Correlation Matrix of Features')
    plt.show()

    # High positive Correlation
    # (tcp_MinWinSz, anon_name) : 0.707364  TCP minimum window size
    # (tcp_AveWinSz, anon_name) : 0.690260  TCP average window size
    # (time_first, layer4_proto) : 0.677882, (time_last, layer4_proto): 0.677884
    # (numPktsSnt, numPktsRcvd) : 0.689236  Number of transmitted packets, Number of received packets
    # (tcp_MSS, anon_name) : 0.654011 TCP maximum segment size
    # (tcp_InitWinSz, anon_name) : 0.596679 TCP initial window size
    # (maxPktSz, anon_name): 0.404485  Maximum layer3 packet size

    # High negative correlation
    # (tcp_RTTAckTripMin, anon_name) : -0.728127 TCP Ack Trip minimum (round trip time between sender and receiver)
    # (layer4_proto, anon_name) : -0.686408
    # (connSrc, anon_name) : -0.433388  Number of connections from source IP to different hosts
    # (connDst, anon_name) : -0.446179  Number of connections from destination IP to different hosts

    plot = df[["tcp_AveWinSz", "tcp_MSS", "tcp_InitWinSz", "skew_iat",
                 "tcp_AveWinSz", "duration", "tcp_RTTAckTripMin", "layer4_proto", "connDst", "connSrc",
                 "minPktSz"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(plot, xticklabels=plot.columns.values, yticklabels=plot.columns.values, annot=True)
    ax.set_title('Correlation Matrix of Features')

    plt.show()

    print("\nSaving model...")
    torch.save(net, "./net.pt")
