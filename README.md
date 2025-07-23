# Fine-Tuning Federated Learning-Based Intrusion Detection Systems for Transportation IoT

This project implements a hybrid Federated Learning (FL) framework for intrusion detection in transportation systems using [Flower](https://github.com/adap/flower), [PyTorch](https://pytorch.org), and the [NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd). It is designed for experimentation on both server-class machines and edge devices such as Raspberry Pi 4s.

---

## Abstract

The rapid advancement of machine learning (ML) and on-device computing has revolutionized the development of Connected and Autonomous Vehicles (CAVs) and Intelligent Transportation Systems (ITS). These systems enhance safety and traffic efficiency but also introduce cybersecurity risks. Federated Learning (FL) enables decentralized model training across edge devices without sharing sensitive data. However, FL deployment in CAV networks faces resource limitations and scalability constraints.

This work proposes a **hybrid server-edge FL framework** where a central server performs model pretraining and edge devices conduct lightweight fine-tuning. This approach reduces memory usage by 42%, cuts training time by 75%, and achieves intrusion detection accuracy up to 99.2%.

---

## Methodology

The project implements a **Federated Learning-based Intrusion Detection System (FL-IDS)** using a 1D Convolutional Neural Network trained on the NSL-KDD dataset. Two training configurations are supported, both leveraging the Federated Averaging (FedAvg) algorithm:

### 1. Full Federated Training (End-to-End FL)
- The **entire model**, including feature extractor and classification head, is distributed to clients.
- Each client trains the **full model** on its local data and sends updated parameters to the server.
- The server uses **FedAvg** to aggregate full model updates.
- Suitable for clients with enough compute and memory to train all layers.

### 2. Federated Fine-Tuning (FedFT)
- The **server pretrains the feature extractor** (e.g., convolutional layers) on proxy data.
- Only the **classification head** is distributed to and trained by clients.
- Clients perform **local fine-tuning** on the head using their own data.
- FedAvg is applied **only to the head**, keeping the frozen backbone unchanged.
- Optimized for edge devices like Raspberry Pi 4s with limited resources.

This hybrid setup enables flexible deployment in real-world CAV/IoT environments where computation and bandwidth are constrained.

---

## Quick Start

### 1. Clone and Set Up

```bash
git clone https://github.com/rlakinie/FL-IDS.git
cd FL-IDS-main
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> A `requirements.txt` is included with all core dependencies.

### 2. Download the NSL-KDD Dataset

- [Kaggle NSL-KDD Link](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Place the following files in the `data/` directory:
  - `KDDTrain+.txt`
  - `KDDTest+.txt`

Ensure these files are structured for use in `utils/load_data.py`.

---

## Running the Project

###  Full Federated Training (End-to-End)

```bash
# Terminal 1: Run server
python server.py

# Terminal 2+: Run clients
python client.py
```

Each client trains all model layers and participates in full-model aggregation.

---

### Federated Fine-Tuning (FedFT: Head Only)

```bash
# Terminal 1: Pretrain model and launch server
python server_FF.py

# Terminal 2+: Run clients (fine-tuning head only)
python client_FF.py
```

Clients only train the classification head. The backbone remains frozen, reducing client-side compute and communication overhead.

---

### Centralized Baseline (No FL)

```bash
python centralized.py
```

Used for comparing federated vs centralized performance.

---

## Project Structure

```
FL-IDS-main/
├── client.py               # Federated client (full model)
├── server.py               # Federated server (full model)
├── client_FF.py            # Client fine-tunes classification head
├── server_FF.py            # Server pretrains CNN backbone
├── centralized.py          # Centralized model (no FL)
├── client_data.py          # Data loader for clients
├── serverFF_data_proxy.py  # Data loader for server pretraining
├── utils/
│   ├── model.py            # 1D CNN model definition
│   ├── load_data.py        # Loads and preprocesses NSL-KDD
│   └── mic.py              # Optional: mutual information ranking
├── data/                   # Place NSL-KDD files here
└── requirements.txt        # Dependencies
```

---

## Future Enhancements

- Incorporate adversarial robustness (e.g., model poisoning, label flipping)
- Support vehicle datasets like HCRL 
- Add Docker containerization for deployment



---

## Reference

This implementation supports experiments from the corresponding research paper:

> Fine-Tuning Federated Learning-based Intrusion Detection Systems for Transportation IoT  
> [IEEE Xplore Link](https://ieeexplore.ieee.org/document/10971473)

---
