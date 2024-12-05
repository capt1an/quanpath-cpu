#pragma once

#include "omsim.h"

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits);

/**
 * @brief Conduct SVSim for gate on single qubit
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qidx the index of target qubit
 */
void SVSimForSingleQubit(Matrix<DTYPE> &gateMatrix, Matrix<DTYPE> &localSv, int qidx);

/**
 * @brief Conduct SVSim for gate on two qubits
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qlow low index of target qubit
 * @param qhigh high index of target qubit
 */
void SVSimForTwoQubit(Matrix<DTYPE> &gateMatrix, Matrix<DTYPE> &localSv, int qlow, int qhigh);

/**
 * @brief [TODO] Conduct the final merge operation in QuanPath
 *
 * @param sv the state vector
 * @param ptrOpmat the pointer to the high-order operation matrix
 */
void merge(Matrix<DTYPE> &sv, Matrix<DTYPE> &ptrOpmat);

void quanpath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numHighQubits, int numLowQubits);