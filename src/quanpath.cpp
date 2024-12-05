#include "quanpath.h"

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits)
{
    int numLowQubits = qc.numQubits - numHighQubits;
    Matrix<DTYPE> opmat, levelmat;
    opmat.identity(1 << numHighQubits);
    levelmat.identity(2);
    for (int j = 0; j < qc.numDepths; ++j)
    {
        int qid = qc.numQubits - 1;

        // get the highest gate matrix
        while (qc.gates[j][qid].isMARK() && qc.gates[j][qid].targetQubits[0] >= numLowQubits)
        {
            // Skip the pseudo placeholder MARK gates placed at control positions
            // when the target gate is applied to high-order qubits
            // If the target gate is applied to low-order qubits, MARK should be regarded as IDE
            --qid;
        }
        // [TODO] Calculate the operation matrix for gates applied to high-order qubits
        // [HINT] We have modified getCompleteMatrix to deal with MARK
        //        In this assignment, MARK is associated with an identity matrix
        // cout << "[TODO] Calculate the operation matrix for gates applied to high-order qubits" << endl;
        // MPI_Abort(MPI_COMM_WORLD, 1);
        levelmat = move(getCompleteMatrix(qc.gates[j][qid]));
        for (int i = qid - 1; i >= numLowQubits; --i)
        {
            if (qc.gates[j][i].isMARK() && qc.gates[j][i].targetQubits[0] >= numLowQubits)
            {
                continue;
            }
            Matrix<DTYPE> tmpmat = move(getCompleteMatrix(qc.gates[j][i]));
            levelmat = move(levelmat.tensorProduct(tmpmat));
        }
        opmat = move(levelmat * opmat);
        // ///////////////////////////////////////////////////////////////////////////
    }
    return opmat;
}

/**
 * @brief Conduct SVSim for gate on single qubit
 *
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qidx the index of target qubit
 */
void SVSimForSingleQubit(Matrix<DTYPE> &gateMatrix, Matrix<DTYPE> &localSv, int qidx)
{
    for (int i = 0; i < localSv.row; i += (1 << (qidx + 1)))
        for (int j = 0; j < (1 << qidx); j++)
        {
            int p = i | j;
            DTYPE q0 = localSv.data[p][0];
            DTYPE q1 = localSv.data[p | 1 << qidx][0];
            localSv.data[p][0] = (gateMatrix.data[0][0] * q0) + (gateMatrix.data[0][1] * q1);
            localSv.data[p | 1 << qidx][0] = (gateMatrix.data[1][0] * q0) + (gateMatrix.data[1][1] * q1);
        }
}

/**
 * @brief Conduct SVSim for gate on two qubits
 *
 * @param gateMatrix the gate matrix pointer
 * @param numQubits the number of low-order qubits
 * @param qlow low index of target qubit
 * @param qhigh high index of target qubit
 */
void SVSimForTwoQubit(Matrix<DTYPE> &gateMatrix, Matrix<DTYPE> &localSv, int qlow, int qhigh)
{
    for (int i = 0; i < localSv.row; i += (1 << (qhigh + 1)))
        for (int j = 0; j < (1 << qhigh); j += 1 << (qlow + 1))
            for (int k = 0; k < (1 << qlow); k++)
            {
                int p = i | j | k;
                DTYPE q0 = localSv.data[p][0];
                DTYPE q1 = localSv.data[p | 1 << qlow][0];
                DTYPE q2 = localSv.data[p | 1 << qhigh][0];
                DTYPE q3 = localSv.data[p | 1 << qlow | 1 << qhigh][0];
                localSv.data[p][0] = gateMatrix.data[0][0] * q0 + gateMatrix.data[0][1] * q1 + gateMatrix.data[0][2] * q2 + gateMatrix.data[0][3] * q3;
                localSv.data[p | 1 << qlow][0] = gateMatrix.data[1][0] * q0 + gateMatrix.data[1][1] * q1 + gateMatrix.data[1][2] * q2 + gateMatrix.data[1][3] * q3;
                localSv.data[p | 1 << qhigh][0] = gateMatrix.data[2][0] * q0 + gateMatrix.data[2][1] * q1 + gateMatrix.data[2][2] * q2 + gateMatrix.data[2][3] * q3;
                localSv.data[p | 1 << qlow | 1 << qhigh][0] = gateMatrix.data[3][0] * q0 + gateMatrix.data[3][1] * q1 + gateMatrix.data[3][2] * q2 + gateMatrix.data[3][3] * q3;
            }
}

/**
 * @brief [TODO] Conduct the final merge operation in QuanPath
 *
 * @param sv the state vector
 * @param ptrOpmat high-order operation matrix
 */
void merge(Matrix<DTYPE> &sv, Matrix<DTYPE> &ptrOpmat)
{
    Matrix<DTYPE> temp = sv;
    for (int i = 0; i < sv.row; i++)
    {
        DTYPE ans = {0, 0};
        for (ll j = 0; j < ptrOpmat.col; j++)
        {
            ans += ptrOpmat.data[i / (sv.row / ptrOpmat.col)][j] * temp.data[i % (sv.row / ptrOpmat.col) + j * (sv.row / ptrOpmat.col)][0];
        }
        sv.data[i][0] = ans;
    }
}

void quanpath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numHighQubits, int numLowQubits)
{
    // Step 1. Calculate the high-order operation matrix in cpu
    Matrix<DTYPE> Opmat = highOMSim(qc, numHighQubits);

    // Step 2. Local SVSim for gates on low-order qubits
    for (int lev = 0; lev < qc.numDepths; ++lev)
    {
        for (int qid = 0; qid < numLowQubits; ++qid)
        {
            QGate &gate = qc.gates[lev][qid];
            if (gate.isIDE() || gate.isMARK())
            {
                continue;
            }
            Matrix<DTYPE> gateMatrix = getCompleteMatrix(gate);

            if (gate.isSingle())
                SVSimForSingleQubit(gateMatrix, hostSv, gate.targetQubits[0]);
            else if (gate.numControls() != 0)
            {
                int q0 = gate.controlQubits[0], q1 = gate.targetQubits[0];
                SVSimForTwoQubit(gateMatrix, hostSv, min(q0, q1), max(q0, q1));
            }
            else
            {
                int q0 = gate.targetQubits[0], q1 = gate.targetQubits[1];
                SVSimForTwoQubit(gateMatrix, hostSv, min(q0, q1), max(q0, q1));
            }
        }
    }
    // printf("Local SVSim for gates on low-order qubits and merge\n");

    // Step 3. Final merge that requires communication
    merge(hostSv, Opmat);
}