import argparse
import os
import json
import shutil
from resnet import setup_seed, ResNet, Reconstruction_autoencoder,Conversion_autoencoder,Speaker_classifier
from loss import *
from dataset import ASVspoof2019
from dataset_with_identity import ASVspoof2019_multi_speaker
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.FloatTensor)


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    # parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data/users/yangli/AIR-ASVspoof-master/LAfeatures/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/users/yangli/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True,
                        default='./models/Re-ocsoftmax_ad/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=6, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--continue_training', action='store_true',
                        help="continue training with previously trained model")
    parser.add_argument('--S1', action='store_true', help="Assist by bonafide speech reconstruction.")
    parser.add_argument('--S2', action='store_true', help="Assist by spoofing voice conversion.")
    parser.add_argument('--S3', action='store_true', help="Assist by speaker classification.")
    parser.add_argument('--dropout1d', action='store_true', help="1D dropout for resnet")
    parser.add_argument('--dropout2d', action='store_true', help="2D dropout for resnet")
    parser.add_argument('--p', type=float, default=0, help="dropout rate for resnet")
    parser.add_argument('--delta', type=float, default=1, help="Factor for controlling the coverage of CA.")
    parser.add_argument('--lambda_r', type=float, default=0.04, help="Trade-off coefficient for bonafide speech reconstruction.")
    parser.add_argument('--lambda_c', type=float, default=1, help="Trade-off coefficient for spoofing voice conversion.")
    parser.add_argument('--lambda_m', type=float, default=0.01, help="Trade-off coefficient for speaker classification.")
    args = parser.parse_args()

    setup_seed(args.seed)
    # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    elif not args.continue_training:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

    # Path for input data
    # assert os.path.exists(args.path_to_database)
    assert os.path.exists(args.path_to_features)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda:0")
    # if int(args.gpu) == 5:
    #     args.device = torch.device("cpu")

    return args


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    id = [0,1]

    lfcc_model = ResNet(3, args.enc_dim, resnet_type='34', nclasses=2, dropout1d=args.dropout1d,
                        dropout2d=args.dropout2d, p=args.p).to(args.device)
    lfcc_model = torch.nn.DataParallel(lfcc_model, device_ids=id)
    if args.S1:
        inverse_model = Reconstruction_autoencoder(args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
        inverse_model = torch.nn.DataParallel(inverse_model, device_ids=id)
    if args.S2:
        CA = Conversion_autoencoder(3, args.enc_dim).to(args.device)
        CA = torch.nn.DataParallel(CA, device_ids=id)
    if args.S3:
        norm_classifier = Speaker_classifier(args.enc_dim,20).to(args.device)
        norm_classifier = torch.nn.DataParallel(norm_classifier, device_ids=id)
    if args.continue_training:
        lfcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt')).to(args.device)
        if args.S1:
            inverse_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_inverse_model.pt')).to(args.device)

    if args.S1:
        lfcc_optimizer = torch.optim.Adam([{'params': lfcc_model.parameters()}, {'params': inverse_model.parameters()}],
                                          lr=args.lr,
                                          betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    else:
        lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=args.lr,
                                          betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    if args.S2:
        CA_optimizer = torch.optim.Adam(CA.parameters(), lr=args.lr,
                                          betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    if args.S3:
        classifier_optimizer =  torch.optim.Adam(norm_classifier.parameters(), lr=args.lr,
                                          betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    training_set = ASVspoof2019_multi_speaker(args.access_type, args.path_to_features, args.path_to_protocol, 'train',
                                'LFCC', feat_len=args.feat_len, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'dev',
                                  'LFCC', feat_len=args.feat_len, padding=args.padding)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=validation_set.collate_fn)

    # feat, _, _, _ = training_set[29]
    # print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss()

    if args.add_loss == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, args.enc_dim, s=args.alpha, m=args.r_real).to(args.device)
        amsoftmax_loss.train()
        amsoftmax_optimzer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)

    if args.add_loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        lfcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, lfcc_optimizer, epoch_num)
        if args.add_loss == "ocsoftmax":
            adjust_learning_rate(args, ocsoftmax_optimzer, epoch_num)
        elif args.add_loss == "amsoftmax":
            adjust_learning_rate(args, amsoftmax_optimzer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (lfcc, audio_fn, tags, labels, speaker) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(args.device)
            labels = labels.to(args.device)
            feats, lfcc_outputs = lfcc_model(lfcc)
            loss1 = criterion(lfcc_outputs, labels)
            if args.S3:
                speaker = speaker.to(args.device)
                index = (labels == 1).nonzero().squeeze()
                feature_select = feats.index_select(0, index)
                label_select_for_classify = speaker.index_select(0,index)
                classifier_optimizer.zero_grad()
                classify_result = norm_classifier(feature_select)
                loss_speaker_norm = criterion(classify_result,label_select_for_classify)
                # print(label_select_for_classify)
                loss_speaker_norm = loss_speaker_norm*args.lambda_m


            if args.S2 and args.add_loss == "softmax":
                index = (labels == 0).nonzero().squeeze()
                lfcc_select = lfcc.index_select(0,index)
                re_ad = CA(lfcc_select)
                feats_ad, lfcc_outputs_ad = lfcc_model(re_ad)
                loss_ad = criterion(lfcc_outputs_ad, labels.index_select(0, index))
                lfcc_loss = loss_ad*args.lambda_c
                if args.S3:
                    lfcc_loss = lfcc_loss+loss_speaker_norm
            else:
                lfcc_loss=0
                if args.S3:
                    lfcc_loss = lfcc_loss+loss_speaker_norm

            if args.S1:
                index = (labels == 1).nonzero().squeeze()
                feats_select = feats.index_select(0, index)
                lfcc_re = inverse_model(feats_select)
                # reconstruction_loss = torch.nn.L1Loss()
                reconstruction_loss = torch.nn.MSELoss()
                lfcc_ = lfcc.index_select(0, index)
                loss2 = reconstruction_loss(lfcc_, lfcc_re) / len(index) * len(lfcc) * args.lambda_r
                lfcc_loss = loss1 + loss2+lfcc_loss
            else:
                lfcc_loss = loss1+lfcc_loss
            if args.add_loss == "softmax":
                lfcc_optimizer.zero_grad()
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()

            if args.add_loss == "ocsoftmax":
                if args.S2:
                    index = (labels == 0).nonzero().squeeze()
                    lfcc_select = lfcc.index_select(0, index)
                    re_ad = CA(lfcc_select)
                    feats_ad, lfcc_outputs_ad = lfcc_model(re_ad)
                    ocsoftmaxloss_adv, _ = ocsoftmax(feats_ad, labels.index_select(0, index))
                    lfcc_loss = ocsoftmaxloss_adv * args.weight_loss*args.lambda_c
                    if args.S3:
                        lfcc_loss = lfcc_loss + loss_speaker_norm
                else:
                    lfcc_loss = 0
                    if args.S3:
                        lfcc_loss = lfcc_loss + loss_speaker_norm


                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                loss1 = ocsoftmaxloss * args.weight_loss
                lfcc_loss = loss1 + loss2+lfcc_loss if args.S1 else loss1+lfcc_loss
                lfcc_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(ocsoftmaxloss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()
                ocsoftmax_optimzer.step()

            if args.add_loss == "amsoftmax":
                if args.S2:
                    index = (labels == 0).nonzero().squeeze()
                    lfcc_select = lfcc.index_select(0,index)
                    re_ad = CA(lfcc_select)
                    feats_ad, lfcc_outputs_ad = lfcc_model(re_ad)
                    outputs_adv, moutputs_adv = amsoftmax_loss(feats_ad, labels.index_select(0, index))
                    lfcc_loss = criterion(moutputs_adv, labels.index_select(0, index))*args.lambda_c
                    if args.S3:
                        lfcc_loss = lfcc_loss + loss_speaker_norm
                else:
                    lfcc_loss = 0
                    if args.S3:
                        lfcc_loss = lfcc_loss + loss_speaker_norm
                outputs, moutputs = amsoftmax_loss(feats, labels)
                loss1 = criterion(moutputs, labels)
                lfcc_loss = loss1 + loss2+lfcc_loss if args.S1 else loss1+lfcc_loss
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_optimizer.zero_grad()
                amsoftmax_optimzer.zero_grad()
                lfcc_loss.backward()
                lfcc_optimizer.step()
                amsoftmax_optimzer.step()

            if args.S3:
                classifier_optimizer.step()
            if args.S2:
                alpha = args.delta
                re_con_struct =  torch.nn.MSELoss()
                index = (labels == 0).nonzero().squeeze()
                lfcc_select = lfcc.index_select(0,index)
                re_ad = CA(lfcc_select)
                feats_ad, lfcc_outputs_ad = lfcc_model(re_ad)
                loss_recon = re_con_struct(re_ad, lfcc_select)
                # loss_ad = criterion(lfcc_outputs_ad, labels.index_select(0, index).fill_(1))
                if args.add_loss == "softmax":
                    loss_ad = criterion(lfcc_outputs_ad, labels.index_select(0, index).fill_(1))
                if args.add_loss == "ocsoftmax":
                    loss_ad,_ = ocsoftmax(feats_ad, labels.index_select(0, index).fill_(1))
                if args.add_loss == "amsoftmax":
                    outputs_ad, moutputs_adv_G = amsoftmax_loss(feats_ad, labels.index_select(0, index).fill_(1))
                    loss_ad = criterion(moutputs_adv_G, labels.index_select(0, index).fill_(1))
                loss_for_G = alpha*loss_ad+loss_recon
                CA_optimizer.zero_grad()
                loss_for_G.backward()
                CA_optimizer.step()
            if (i + 1) % 50 == 0:
                print("[EPOCH: %3d] [TOTAL_LOSS: %.3f]\n" % (epoch_num + 1, lfcc_loss))

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(np.nanmean(trainlossDict[monitor_loss])) + "\n")
        torch.save(lfcc_model, os.path.join(args.out_fold, 'checkpoint','anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        torch.save(CA, os.path.join(args.out_fold, 'checkpoint','anti-spoofing_CA_%d.pt' % (epoch_num + 1)))
        # Val the model
        lfcc_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                labels = labels.to(args.device)

                feats, lfcc_outputs = lfcc_model(lfcc)

                lfcc_loss = criterion(lfcc_outputs, labels)
                score = F.softmax(lfcc_outputs, dim=1)[:, 0]

                if args.add_loss == "softmax":
                    devlossDict["softmax"].append(lfcc_loss.item())
                elif args.add_loss == "amsoftmax":
                    outputs, moutputs = amsoftmax_loss(feats, labels)
                    lfcc_loss = criterion(moutputs, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]
                    devlossDict[args.add_loss].append(lfcc_loss.item())
                elif args.add_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args.add_loss].append(ocsoftmaxloss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_val_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            val_eer = min(val_eer, other_val_eer)

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) + "\n")
            print("Val EER: {}".format(val_eer))

        torch.save(lfcc_model, os.path.join(args.out_fold, 'checkpoint',
                                            'anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        if args.S1:
            torch.save(inverse_model, os.path.join(args.out_fold, 'checkpoint',
                                                   'anti-spoofing_inverse_model_%d.pt' % (epoch_num + 1)))
        if args.add_loss == "ocsoftmax":
            loss_model = ocsoftmax
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        elif args.add_loss == "amsoftmax":
            loss_model = amsoftmax_loss
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        else:
            loss_model = None

        if val_eer < prev_eer:
            # Save the model checkpoint
            torch.save(lfcc_model, os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
            if args.S1:
                torch.save(inverse_model, os.path.join(args.out_fold, 'anti-spoofing_inverse_model.pt'))
            if args.add_loss == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "amsoftmax":
                loss_model = amsoftmax_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break

    return lfcc_model, loss_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
    if args.add_loss == "softmax":
        loss_model = None
    else:
        loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))