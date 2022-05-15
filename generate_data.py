import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matlab.engine

eng = matlab.engine.start_matlab()

import random
from scipy import io


def generate_meta(args):

    save_path = args.data_path

    mod = 1  # modulation level
    M = 2**mod
    us_SNR = torch.randint(min(args.snr), max(args.snr) + 1, (args.train_task_num,))
    SNR_rho = (1 / mod) * (10.0 ** (-0.1 * us_SNR))

    ######Generate_Train_Data########
    channel = torch.exp(
        -1 * 2.0 * torch.tensor(list(range(0, args.L, 1)), dtype=torch.double)
    )
    channel = (channel / channel.sum()).flip(0).unsqueeze(0)
    channel_for_train = torch.zeros(args.train_task_num, args.L) + 1j * torch.zeros(
        args.train_task_num, args.L
    )
    channel_for_train = ((channel / 2) ** 0.5) * (
        torch.randn(args.train_task_num, args.L)
        + 1j * torch.randn(args.train_task_num, args.L)
    )

    coded_msg = torch.randint(0, 2, (args.train_task_num, mod * args.sym_length))
    coded_msg_mat = {"codedBits": coded_msg.numpy()}
    io.savemat("codedBits.mat", coded_msg_mat)
    symbol_vec = torch.t(torch.tensor(np.array(eng.modulation(M))))
    symbol_vec = symbol_vec + 1j * torch.zeros_like(symbol_vec)
    symbol_mat = torch.zeros(
        args.train_task_num, symbol_vec.shape[1], args.L
    ) + 1j * torch.zeros(args.train_task_num, symbol_vec.shape[1], args.L)
    for i in range(symbol_vec.shape[1]):
        if i >= args.L - 1:
            symbol_mat[:, i, :] = symbol_vec[:, i - args.L + 1 : i + 1]
        else:
            symbol_mat[:, i, -(i + 1) :] = symbol_vec[:, : (i + 1)]

    Y_train = torch.sum(symbol_mat * channel_for_train.unsqueeze(1), -1, False)
    Y_train = Y_train + ((SNR_rho.unsqueeze(-1) / 2) ** 0.5) * torch.randn_like(Y_train)
    Y_input_train = torch.zeros(
        args.train_task_num, symbol_vec.shape[1], 2 * args.L - 1
    ) + 1j * torch.zeros(args.train_task_num, symbol_vec.shape[1], 2 * args.L - 1)
    for i in range(Y_input_train.shape[1]):
        if i < (args.L - 1):
            Y_input_train[:, i, (args.L - 1 - i) :] = Y_train[:, : (i + args.L)]
        elif i + args.L > (Y_input_train.shape[1]):
            Y_input_train[:, i, : (Y_input_train.shape[1] + args.L - i - 1)] = Y_train[
                :, (i - args.L + 1) :
            ]
        else:
            Y_input_train[:, i, :] = Y_train[:, (i - args.L + 1) : (i + args.L)]

    if args.ce_type == 1:
        train_channel_for_loss = channel_for_train.unsqueeze(1).repeat(
            1, args.sym_length, 1
        )

        train_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
            args.train_task_num, args.sym_length, args.snr, args.L
        )
    elif args.ce_type == 2:
        H_hat = (
            channel_for_train
            + ((args.noise / 2) ** 0.5)
            * torch.randn_like(channel_for_train)
            * channel_for_train
        )
        train_channel_for_loss = H_hat.unsqueeze(1).repeat(1, args.sym_length, 1)

        train_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
            args.noise, args.train_task_num, args.sym_length, args.snr, args.L
        )
    else:
        raise NotImplementedError

    train_save_name = save_path + "train/" + train_save_arg
    torch.save(
        {
            "Y_train": Y_train,  # [args.train_task_num, args.sym_length]
            "Y_input_train": Y_input_train,  # [args.train_task_num, args.sym_length, 2*args.L-1]
            "original_x": coded_msg,  # [args.train_task_num, args.sym_length]
            "state_x": symbol_mat,  # [args.train_task_num, args.sym_length]
            "channel": train_channel_for_loss,  # [args.train_task_num, args.sym_length, args.L]
        },
        train_save_name,
    )
    print("train_data_saved")

    ######Generate_val_Data########
    us_SNR = torch.randint(min(args.snr), max(args.snr) + 1, (args.val_task_num,))
    SNR_rho = (1 / mod) * (10.0 ** (-0.1 * us_SNR))

    channel = torch.exp(
        -1 * 2.0 * torch.tensor(list(range(0, args.L, 1)), dtype=torch.double)
    )
    channel = (channel / channel.sum()).flip(0).unsqueeze(0)
    channel_for_val = torch.zeros(args.val_task_num, args.L) + 1j * torch.zeros(
        args.val_task_num, args.L
    )
    channel_for_val = ((channel / 2) ** 0.5) * (
        torch.randn(args.val_task_num, args.L)
        + 1j * torch.randn(args.val_task_num, args.L)
    )

    coded_msg = torch.randint(0, 2, (args.val_task_num, mod * args.sym_length))
    coded_msg_mat = {"codedBits": coded_msg.numpy()}
    io.savemat("codedBits.mat", coded_msg_mat)
    symbol_vec = torch.t(torch.tensor(np.array(eng.modulation(M))))
    symbol_vec = symbol_vec + 1j * torch.zeros_like(symbol_vec)
    symbol_mat = torch.zeros(
        args.val_task_num, symbol_vec.shape[1], args.L
    ) + 1j * torch.zeros(args.val_task_num, symbol_vec.shape[1], args.L)
    for i in range(symbol_vec.shape[1]):
        if i >= args.L - 1:
            symbol_mat[:, i, :] = symbol_vec[:, i - args.L + 1 : i + 1]
        else:
            symbol_mat[:, i, -(i + 1) :] = symbol_vec[:, : (i + 1)]

    Y_val = torch.sum(symbol_mat * channel_for_val.unsqueeze(1), -1, False)
    Y_val = Y_val + ((SNR_rho.unsqueeze(-1) / 2) ** 0.5) * torch.randn_like(Y_val)
    Y_input_val = torch.zeros(
        args.val_task_num, symbol_vec.shape[1], 2 * args.L - 1
    ) + 1j * torch.zeros(args.val_task_num, symbol_vec.shape[1], 2 * args.L - 1)
    for i in range(Y_input_val.shape[1]):
        if i < (args.L - 1):
            Y_input_val[:, i, (args.L - 1 - i) :] = Y_val[:, : (i + args.L)]
        elif i + args.L > (Y_input_val.shape[1]):
            Y_input_val[:, i, : (Y_input_val.shape[1] + args.L - i - 1)] = Y_val[
                :, (i - args.L + 1) :
            ]
        else:
            Y_input_val[:, i, :] = Y_val[:, (i - args.L + 1) : (i + args.L)]

    if args.ce_type == 1:
        val_channel_for_loss = channel_for_val.unsqueeze(1).repeat(
            1, args.sym_length, 1
        )

        val_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
            args.val_task_num, args.sym_length, args.snr, args.L
        )
    elif args.ce_type == 2:
        H_hat = (
            channel_for_val
            + ((args.noise / 2) ** 0.5)
            * torch.randn_like(channel_for_val)
            * channel_for_val
        )
        val_channel_for_loss = H_hat.unsqueeze(1).repeat(1, args.sym_length, 1)

        val_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
            args.noise, args.val_task_num, args.sym_length, args.snr, args.L
        )
    else:
        raise NotImplementedError

    val_save_name = save_path + "val/" + val_save_arg
    torch.save(
        {
            "Y_val": Y_val,  # [args.val_task_num, args.sym_length]
            "Y_input_val": Y_input_val,  # [args.val_task_num, args.sym_length, 2*args.L-1]
            "original_x": coded_msg,  # [args.val_task_num, args.sym_length]
            "state_x": symbol_mat,  # [args.val_task_num, args.sym_length]
            "channel": val_channel_for_loss,  # [args.val_task_num, args.sym_length, args.L]
        },
        val_save_name,
    )
    print("val_data_saved")

    ######Generate_test_Data########
    channel = torch.exp(
        -1 * 2.0 * torch.tensor(list(range(0, args.L, 1)), dtype=torch.double)
    )
    channel = (channel / channel.sum()).flip(0).unsqueeze(0)
    channel_for_test = torch.zeros(args.test_task_num, args.L) + 1j * torch.zeros(
        args.test_task_num, args.L
    )
    channel_for_test = ((channel / 2) ** 0.5) * (
        torch.randn(args.test_task_num, args.L)
        + 1j * torch.randn(args.test_task_num, args.L)
    )

    coded_msg = torch.randint(0, 2, (args.test_task_num, mod * args.sym_length))
    coded_msg_mat = {"codedBits": coded_msg.numpy()}
    io.savemat("codedBits.mat", coded_msg_mat)
    symbol_vec = torch.t(torch.tensor(np.array(eng.modulation(M))))
    symbol_vec = symbol_vec + 1j * torch.zeros_like(symbol_vec)
    symbol_mat = torch.zeros(
        args.test_task_num, symbol_vec.shape[1], args.L
    ) + 1j * torch.zeros(args.test_task_num, symbol_vec.shape[1], args.L)
    for i in range(symbol_vec.shape[1]):
        if i >= args.L - 1:
            symbol_mat[:, i, :] = symbol_vec[:, i - args.L + 1 : i + 1]
        else:
            symbol_mat[:, i, -(i + 1) :] = symbol_vec[:, : (i + 1)]

    for snr in args.snr_test:
        SNR_rho = (1 / mod) * (10.0 ** (-0.1 * torch.tensor(snr)))

        Y_test = torch.sum(symbol_mat * channel_for_test.unsqueeze(1), -1, False)
        Y_test = Y_test + ((SNR_rho / 2) ** 0.5) * torch.randn_like(Y_test)
        Y_input_test = torch.zeros(
            args.test_task_num, symbol_vec.shape[1], 2 * args.L - 1
        ) + 1j * torch.zeros(args.test_task_num, symbol_vec.shape[1], 2 * args.L - 1)
        for i in range(Y_input_test.shape[1]):
            if i < (args.L - 1):
                Y_input_test[:, i, (args.L - 1 - i) :] = Y_test[:, : (i + args.L)]
            elif i + args.L > (Y_input_test.shape[1]):
                Y_input_test[:, i, : (Y_input_test.shape[1] + args.L - i - 1)] = Y_test[
                    :, (i - args.L + 1) :
                ]
            else:
                Y_input_test[:, i, :] = Y_test[:, (i - args.L + 1) : (i + args.L)]

        if args.ce_type == 1:
            test_channel_for_loss = channel_for_test.unsqueeze(1).repeat(
                1, args.sym_length, 1
            )

            test_save_arg = "meta_H1_{}x{}_snr{}_L{}.pt".format(
                args.test_task_num, args.sym_length, snr, args.L
            )
        elif args.ce_type == 2:
            H_hat = (
                channel_for_test
                + ((args.noise / 2) ** 0.5)
                * torch.randn_like(channel_for_test)
                * channel_for_test
            )
            test_channel_for_loss = H_hat.unsqueeze(1).repeat(1, args.sym_length, 1)

            test_save_arg = "meta_H2_{}_{}x{}_snr{}_L{}.pt".format(
                args.noise, args.test_task_num, args.sym_length, snr, args.L
            )
        else:
            raise NotImplementedError

        test_save_name = save_path + "test/" + test_save_arg
        torch.save(
            {
                "Y_test": Y_test,  # [args.test_task_num, args.sym_length]
                "Y_input_test": Y_input_test,  # [args.test_task_num, args.sym_length, 3*args.L-1]
                "original_x": coded_msg,  # [args.test_task_num, args.sym_length]
                "state_x": symbol_mat,  # [args.test_task_num, args.sym_length]
                "channel": test_channel_for_loss,  # [args.test_task_num, args.L]
            },
            test_save_name,
        )
        print("test_data_saved at snr", snr)
