import torch.nn as nn
import torch.optim as optim
import configargparse
import os
import torch
import dataloader
from model import MLP
from tqdm import tqdm
import torch.nn.functional as F
import neptune.new as neptune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args(args=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--disable_neptune', action='store_true')

    parser.add_argument('--log_dir', help='the directory where all logs are saved', type=str, default='./logs')
    parser.add_argument('--data_dir', help='where the data is', type=str, default='../data/datasets/results/normalized_numpy')

    parser.add_argument('--measure_accuracy_epoch', type=int, default=3)
    parser.add_argument('--save_model_epoch', type=int, default=200)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--expname', help='experiment name', type=str)

    parser.add_argument("--netdepth", type=int, default=4,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_p', type=float, default=0.5)

    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--channels', nargs='+', type=int)
    return parser.parse_args(args)


def save_network(args, model:torch.nn.Module, epoch):
    path = os.path.join(args.log_dir, args.expname, f'{epoch}.pt')
    torch.save(model.state_dict(), path)


def load_model(args, model):
    # find most up to date save point
    epochs = [int(l[:l.rfind('.')]) for l in os.listdir(os.path.join(args.log_dir, args.expname))]
    if len(epochs) == 0:
        return model, 0

    # load the last save
    epoch = max(epochs)
    path = os.path.join(args.log_dir, args.expname, f'{epoch}.pt')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, epoch


def test_model_accuracy(model, data_loader):
    correct_chunks = 0
    wrong_chunks = 0
    correct_files = 0
    wrong_files = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = data
            X = X[0]
            X = X.view([X.shape[0], -1]).to(device=device)
            y = y[0].to(device=device)
            y_true.append(y)
            # zero the parameter gradients
            pred = model(X, testing=True)

            pred = torch.argmax(pred, dim=1)
            y_preds.append(pred)
            same = (pred == y)
            correct_chunks += same.sum().item()
            wrong_chunks += same.shape[0]

            file_prediction = torch.mode(pred,0)
            if file_prediction[0] == y[0]:
                correct_files += 1
            else:
                wrong_files += 1

    wrong_chunks -= correct_chunks
    return correct_files / (correct_files + wrong_files), correct_chunks / (correct_chunks + wrong_chunks)

def train(args=None, tqdm_monitor=True):
    args = parse_args(args)
    print(args.config)

    run = None
    params = {
        'netdepth': args.netdepth,
        'netwidth': args.netwidth,
        'dropout': args.dropout,
        'dropout_p': args.dropout_p,
        'learning_rate': args.lr,
        'expname': args.expname,
        'epochs_per_accuracy': args.measure_accuracy_epoch
    }
    if not args.disable_neptune:
        run = neptune.init(
            project="michael.lellouch/neuro",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmVjNDRjMy0wNjQwLTRhOWItODZlZi1mNzAyOTBmNmZjMjUifQ==",
        )  # your credentials
        run['parameters'] = params

    # init datasetes
    train_dataset = dataloader.EEGDataset(os.path.join(args.data_dir, 'train'), device=device, channels=args.channels)
    test_dataset = dataloader.EEGDataset(os.path.join(args.data_dir, 'test'), device=device, channels=args.channels)
    double_dip_dataset = dataloader.EEGDataset(os.path.join(args.data_dir, 'double_dip'), device=device, channels=args.channels)

    number_of_channels = train_dataset[0][0].shape[1]
    number_of_features = train_dataset[0][0].shape[2]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    double_dip_loader = torch.utils.data.DataLoader(double_dip_dataset, batch_size=args.batch_size, shuffle=True)

    # init network
    first_epoch = 0
    model = MLP(depth=args.netdepth, width=args.netwidth, input_channels=number_of_channels * number_of_features, output_ch=len(dataloader.classes.keys()),
                use_dropout=args.dropout, dropout_p=args.dropout_p)
    if os.path.exists(os.path.join(args.log_dir, args.expname)):
        print(args.expname)
        model, first_epoch = load_model(args, model)
    else:
        os.mkdir(os.path.join(args.log_dir, args.expname))

    model = model.to(device=device)
    # init training params
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_classes = len(dataloader.classes.keys())
    best_double_dip = 0
    best_test = 0
    # run training
    iteration = range(first_epoch, args.epochs)
    if tqdm_monitor:
        iteration = tqdm(iteration)
    for epoch in iteration:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = data
            X = X[0]
            X = X.view([X.shape[0], -1]).to(device=device)
            y = y[0].to(device=device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            del data
            del X
            del y
            # print statistics
            running_loss += loss.item()

        if not args.disable_neptune:
            run["train/loss"].log(running_loss)
        if (epoch + 1) % args.measure_accuracy_epoch == 0:
            double_dip_accuracy = test_model_accuracy(model, double_dip_loader)
            test_accuracy = test_model_accuracy(model, test_loader)

            best_double_dip = max(best_double_dip, double_dip_accuracy[0])
            best_test = max(best_test, test_accuracy[0])

            if not args.disable_neptune:
                run['double_dip/file_accuracy'].log(double_dip_accuracy[0])
                run['double_dip/chunk_accuracy'].log(double_dip_accuracy[1])

                run['test/file_accuracy'].log(test_accuracy[0])
                run['test/chunk_accuracy'].log(test_accuracy[1])

        if (epoch + 1) % args.save_model_epoch == 0:
            save_network(args, model, epoch)

    print('Finished Training')
    return best_double_dip, best_test


if __name__ == '__main__':
    train()