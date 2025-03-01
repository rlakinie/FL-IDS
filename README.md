# Fine-Tuning Federated Learning-Based Intrusion Detection Systems for Transportatio IoT
Federated Learning based Intrusion Detection Systems with [Flower](https://github.com/adap/flower), Raspberry Pi 4s and the [NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)

## Overview

This repository represents a basic implementation for the corresponding research paper, Fine-Tuning Federated Learning-based Intrusion Detection Systems for Transportation IoT, whcich can be found [here]() once its published. Arxiv preprint [here](https://arxiv.org/abs/2502.06099).

## Abstract
The rapid advancement of machine learning (ML) and on-device computing has revolutionized various industries, including transportation, through the development of Connected and Autonomous Vehicles (CAVs) and Intelligent Transportation Systems (ITS). These technologies improve traffic management and vehicle safety, but also introduce significant security and privacy concerns, such as cyberattacks and data breaches. Traditional Intrusion Detection Systems (IDS) are increasingly inadequate in detecting modern threats, leading to the adoption of ML-based IDS solutions. Federated Learning (FL) has emerged as a promising method for enabling the decentralized training of IDS models on distributed edge devices without sharing sensitive data. However, deploying FL-based IDS in CAV networks poses unique challenges, including limited computational and memory resources on edge devices, competing demands from critical applications such as navigation and safety systems, and the need to scale across diverse hardware and connectivity conditions. To address these issues, we propose a hybrid server-edge FL framework that offloads pre-training to a central server while enabling lightweight fine-tuning on edge devices. This approach reduces memory usage by up to 42%, decreases training times by up to 75%, and achieves competitive IDS accuracy of up to 99.2%. Scalability analysis furtherdemonstrates minimal performance degradation as the number of clients increases, highlighting the framework’s feasibility for CAV networks and other IoT applications.


## Methodology
The project's aim is to propose a hybrid server-edge FL framework that offloads pre-training to a central server while enabling lightweight fine-tuning on edge devices, particularly for connected and autonomous vehicle architecture and other related applications. This implmnetation addressses resource constraints in Federated Learning on the edge, particularly for CAV computing environments. 

Implementation includes training a 1D ConvNet for Network Intrusuion Detection, deployed across a server and a number of Raspberry Pis acting as clients. The server pretrains a global intrusion detection model using proxy data.

## Quick Start Guide
- Download the [NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Place the training and test files in the project's data folder

## 
Project code will be uploaded soon
## Future Enhancements
