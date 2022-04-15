# Raul: on-device training library
Huawei  
Noah`s Ark Lab  
AI enabling engineering team, Moscow  


## 1 General
Raul on-device training library is C++ based software designed to train complex neural network (NN) topologies using minimal external dependencies and minimize ROM/RAM memory footprint. Current implementation of Raul version is CPU-based with OpenBLAS mathematical back-end used for intensive mathematical operations: matrix multiplication and element wise vector operations.

## 2 Layers and topologies
Internal representation of typical NN operations based on directed acyclic graph (DAG). Input, output variables and layer parameters as weights and biases are stored as tensors. Current version of Raul support tensors based on float32 data only. Memory manager controls each tensor allocation and deallocation. Usage of DAG together with managed tensors allows creating complex topologies with residual connections.
To train NN stochastic gradient descent (SGD) algorithm used.

## 3 Cross-platform support
Source code of Raul is C++ written in cross-platform way. To build library (compile or cross-compile) for target platform it is enough to support toolchain (C++ compiler, linker).

## 4 Functional tests
To check correctness of SGD implementation several functional tests implemented. As a baseline for functional test pytorch implementation of SGD used. Topologies for NN created using pytorch as well as initialization of weights and biases. Initial weights and biases has been stored into file for further usage in Raul's functional test. Initial evaluation of NN accuracy performed and followed by training process. Each 100 iterations of training loss function saved into file to use it as a reference. Final evaluation of NN accuracy performed after training is finished.
Typical scenario of Raul`s functional test:
1. Set-up NN topology
2. Load initial weights and biases from files
3. Perform initial evaluation of NN accuracy
4. Compare evaluation result with baseline with some tolerance (tol)
5. Run training procedure for 1 epoch, each 100 iteration loss is compared with baseline
6. Perform final evaluation of NN accuracy
7. Compare evaluation result with baseline with some tolerance