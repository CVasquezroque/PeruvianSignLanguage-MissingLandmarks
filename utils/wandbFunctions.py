#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:34:41 2021

@author: joe
"""
import wandb

def initConfigWandb(num_layers, num_classes, batch_size, 
                    nEpoch, lrn_rate, hidden_size, dropout,
                    weight_decay, epsilon):

    wandb.init(project='demo-psl', entity='joenatan30')
    run = wandb.init()

    config = wandb.config
    
    config["num_layers"] = num_layers
    config["num_classes"] = num_classes
    config.batch_size = batch_size
    config.epochs = nEpoch
    config.learning_rate = lrn_rate
    config["hidden_size"] = hidden_size
    config.dropout = dropout
    config["weight_decay"] = weight_decay
    config["epsilon"] = epsilon

def wandbLog(trainLoss, TrainAcc, testLoss, TestAcc):
    wandb.log({"Train loss": trainLoss,
               "Train accuracy": TrainAcc,
               "Test Loss": testLoss,
               "Test accuracy": TestAcc
               })
    
def watch(model):
    wandb.watch(model)
    
def finishWandb():
    wandb.finish()
    
def sendConfusionMatrix(ground_truth, predictions, class_names, cmTrain=True):
    if(cmTrain):
        wandb.log({"TRAIN_conf_mat" : wandb.plot.confusion_matrix(
            probs=None,
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names,
            title="TRAIN_conf_mat")})
    else:
        wandb.log({"TEST_conf_mat" : wandb.plot.confusion_matrix(
            probs=None,
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names,
            title="TEST_conf_mat")})
            