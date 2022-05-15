from meta import MetaLearner
from naive import Naive

from generate_data import generate_meta

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import io

import random
import argparse


class PilotSignalDataset(Dataset):
    def __init__(self, y, y_vec, h_vec, x, s):
        self.y = y
        self.y_vec = y_vec
        self.h_vec = h_vec
        self.x = x
        self.s = s

    def __len__(self):
        return self.y_vec.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.y_vec[idx], self.h_vec[idx], self.x[idx], self.s[idx]


def main():

    # fix the random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # If you want to generate new data, then generate data with setting args.gen_data as True
    if args.gen_data:
        generate_meta(args)

    save_path = args.data_path

    #
    if args.ce_type == 1:
        train_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
            args.train_task_num, args.sym_length, args.snr, args.L
        )
        val_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
            args.val_task_num, args.sym_length, args.snr, args.L
        )
    elif args.ce_type == 2:
        train_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
            args.noise, args.train_task_num, args.sym_length, args.snr, args.L
        )
        val_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
            args.noise, args.val_task_num, args.sym_length, args.snr, args.L
        )
    else:
        raise NotImplementedError

    loaded_train_data = torch.load(save_path + "train_" + train_save_arg)
    Y_train = loaded_train_data["Y_train"]
    Y_input_train = loaded_train_data["Y_input_train"]
    channel_train = loaded_train_data["channel"]
    x_vec_train = loaded_train_data["original_x"]
    state_train = loaded_train_data["state_x"]

    loaded_val_data = torch.load(save_path + "val_" + val_save_arg)
    Y_val = loaded_val_data["Y_val"]
    Y_input_val = loaded_val_data["Y_input_val"]
    channel_val = loaded_val_data["channel"]
    x_vec_val = loaded_val_data["original_x"]
    state_val = loaded_val_data["state_x"]

    train_data = PilotSignalDataset(
        Y_train, Y_input_train, channel_train, x_vec_train, state_train
    )
    val_data = PilotSignalDataset(Y_val, Y_input_val, channel_val, x_vec_val, state_val)

    input_size = 2 * (2 * args.L - 1)  # multiply 2 for complex number

    meta = MetaLearner(
        Naive,
        (input_size, args),
        meta_batchsz=args.batch_size,
        update_lr=args.update_lr,
        temp_lr=args.temp_lr,
        meta_lr=args.meta_lr,
        gamma=args.gamma,
        num_updates=args.update_step,
        pl=args.P,
        reg=args.reg,
        reg2=args.reg2,
    ).to(device)

    # main loop
    lowest_ser = 1.0
    model_arg = "meta_H{}_snr{}_L{}_".format(args.ce_type, args.snr, args.L)
    model_file_name = args.model_path + model_arg + args.modelname + ".pt"
    for epoch_num in range(args.epoch):

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        loss = []
        loss1 = []
        loss2 = []
        ser = []
        for step, data in enumerate(train_loader):
            y_qry, y_vec_qry, h_qry, x_qry, s_qry = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )

            accs, losses, loss1s, loss2s, scheduler = meta(
                y_qry, y_vec_qry, h_qry, x_qry, s_qry
            )
            ser += list(1.0 - np.array(accs))
            loss += losses
            loss1 += loss1s
            loss2 += loss2s

        cur_sers = []
        for step, data in enumerate(val_loader):
            y_qry, y_vec_qry, h_qry, x_qry, s_qry = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )
            val_acc, losses = meta.pred(y_qry, y_vec_qry, h_qry, x_qry, s_qry)
            cur_sers += list(1.0 - val_acc)
        cur_ser = np.array(cur_sers).mean()
        print(
            "\nepoch:{} \nval_ser={}\ntrain loss={}(loss1={}, loss2={})\n".format(
                epoch_num,
                cur_ser,
                np.array(loss).mean(),
                np.array(loss1).mean(),
                np.array(loss2).mean(),
            )
        )
        if cur_ser < lowest_ser:
            torch.save(meta, model_file_name)
            lowest_ser = cur_ser
            print(
                "model saved at epoch {}, current val_ser = {}".format(
                    epoch_num, cur_ser
                )
            )
        scheduler.step()

    meta = torch.load(model_file_name)

    print("===========test_accuracy============")
    for snr_idx in range(len(args.snr_test)):
        test_sers = []
        snr = args.snr_test[snr_idx]

        if args.ce_type == 1:
            test_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
                args.test_task_num, args.sym_length, args.snr, args.L
            )
        elif args.ce_type == 2:
            test_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
                args.noise, args.test_task_num, args.sym_length, args.snr, args.L
            )
        else:
            raise NotImplementedError

        loaded_test_data = torch.load(save_path + "test_" + test_save_arg)
        Y_test = loaded_test_data["Y_test"]
        Y_input_test = loaded_test_data["Y_input_test"]
        channel_test = loaded_test_data["channel"]
        x_vec_test = loaded_test_data["original_x"]
        state_test = loaded_test_data["state_x"]

        test_data = PilotSignalDataset(
            Y_test, Y_input_test, channel_test, x_vec_test, state_test
        )
        test_loader = DataLoader(
            test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=2
        )

        for i, data in enumerate(test_loader):
            y_qry, y_vec_qry, h_qry, x_qry, s_qry = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )

            test_acc, losses = meta.pred(y_qry, y_vec_qry, h_qry, x_qry, s_qry)

            test_ser = 1.0 - test_acc
            test_sers += list(test_ser)

        print("snr={}: final ser = {}".format(snr, np.array(test_sers).mean()))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    argparser.add_argument("--epoch", type=int, help="epoch number", default=50)
    argparser.add_argument("--batch-size", type=int, help="meta batch size", default=50)
    argparser.add_argument(
        "--test-batch-size", type=int, help="meta batch size for test", default=50
    )
    argparser.add_argument(
        "--meta-lr", type=float, help="meta-level outer learning rate", default=0.01
    )
    argparser.add_argument(
        "--update-lr",
        type=float,
        help="task-level inner learning rate",
        default=0.1,
    )
    argparser.add_argument(
        "--temp-lr",
        type=float,
        help="task-level inner update learning rate for temperature parameters",
        default=0.001,
    )
    argparser.add_argument(
        "--update-step", type=int, help="task-level inner update steps", default=4
    )
    argparser.add_argument(
        "--gamma", type=float, help="gamma value in scheduler", default=0.95
    )
    # additional arguments
    argparser.add_argument(
        "--gen-data", action="store_true", default=False, help="generates new data"
    )
    argparser.add_argument(
        "--ce-type",
        type=int,
        default=1,
        metavar="N",
        help="type of channel estimation (perfect(1), noisy(2))",
    )
    argparser.add_argument(
        "--noise", type=float, default=0.02, help="variance of noise"
    )
    argparser.add_argument(
        "--snr",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="snr list",
    )
    argparser.add_argument(
        "--snr-test",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="test snr list",
    )
    argparser.add_argument(
        "--sym-length",
        type=int,
        help="train data's symbol length per task",
        default=10000,
    )
    argparser.add_argument(
        "--train-task-num",
        type=int,
        default=10000,
        metavar="N",
        help="task number for training",
    )
    argparser.add_argument(
        "--val-task-num",
        type=int,
        default=500,
        metavar="N",
        help="task number for validation",
    )
    argparser.add_argument(
        "--test-task-num",
        type=int,
        default=500,
        metavar="N",
        help="task number for testing",
    )
    argparser.add_argument(
        "--L", type=int, default=4, metavar="N", help="memory length"
    )
    argparser.add_argument(
        "--P", type=int, default=100, metavar="N", help="data length for adaptation"
    )
    argparser.add_argument("--loss-alpha", type=float, default=0.1, help="loss ratio")
    argparser.add_argument(
        "--reg",
        type=float,
        default=0.1,
        help="regularization argument for model in meta loss",
    )
    argparser.add_argument(
        "--reg2",
        type=float,
        default=0.01,
        help="regularization argument for temperature-parametr in update loss",
    )
    argparser.add_argument(
        "--modelname", type=str, default="model", help="model-save-name"
    )
    argparser.add_argument("--data-path", type=str, default="data/", help="data path")
    argparser.add_argument(
        "--model-path", type=str, default="model/", help="model path"
    )

    args = argparser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
