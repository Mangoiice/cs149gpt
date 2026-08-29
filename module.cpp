#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] = val; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for(int b = 0; b < B; ++b)
    {
        for(int h = 0; h < H; ++h)
        {
            // 生成QK^T矩阵的(i, j)元素
            for(int i = 0; i < N; ++i)
            {
                for(int j = 0; j < N; ++j)
                {
                    float val = 0.0;
                    // 遍历Q矩阵的第i行和K矩阵的第j列
                    for(int k = 0; k < d; ++k)
                        val += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    // 写入
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // 对每一行进行softmax
            for(int i = 0; i < N; ++i)
            {
                float lx = 0.0;
                for(int j = 0; j < N; ++j)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = std::exp(val);
                    lx += val;
                    twoDimWrite(QK_t, i, j, N, val);
                }
                for(int j = 0; j < N; ++j)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= lx;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
            
            // QK^T(N * N) * V(N * d)
            for(int i = 0; i < N; ++i)
            {
                for(int j = 0; j < d; ++j)
                {
                    float val = 0.0;
                    // 遍历QK^T矩阵的第i行和V矩阵的第j列
                    for(int k = 0; k < N; ++k)
                        val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                    // 写入
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }

        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    int tileSize = 256;
                    
    for(int b = 0; b < B; ++b)
    {
        for(int h = 0; h < H; ++h)
        {
            // Q(N * d) * K(N * d) -> QK^T(N * N)
            for(int iBlock = 0; iBlock < N; iBlock += tileSize)
            {
                for(int jBlock = 0; jBlock < N; jBlock += tileSize)
                {
                    for(int kBlock = 0; kBlock < d; kBlock += tileSize)
                    {
                        int iUpper = std::min(N, iBlock + tileSize);
                        int jUpper = std::min(N, jBlock + tileSize);
                        int kUpper = std::min(d, kBlock + tileSize);
                        for(int i = iBlock; i < iUpper; ++i)
                        {
                            for(int j = jBlock; j < jUpper; ++j)
                            {
                                float val = twoDimRead(QK_t, i, j, N);
                                for(int k = kBlock; k < kUpper; ++k)  
                                    val += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                                twoDimWrite(QK_t, i, j, N, val);
                            }
                        }
                    } 
                }
            }
            
            // 对每一行进行softmax
            for(int i = 0; i < N; ++i)
            {
                float lx = 0.0;
                for(int j = 0; j < N; ++j)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = std::exp(val);
                    lx += val;
                    twoDimWrite(QK_t, i, j, N, val);
                }
                for(int j = 0; j < N; ++j)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= lx;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
            
            // QK^T(N * N) * V(N * d)
            for(int iBlock = 0; iBlock < N; iBlock += tileSize)
            {
                for(int jBlock = 0; jBlock < d; jBlock += tileSize)
                {
                    for(int kBlock = 0; kBlock < N; kBlock += tileSize)
                    {
                        int iUpper = std::min(N, iBlock + tileSize);
                        int jUpper = std::min(d, jBlock + tileSize);
                        int kUpper = std::min(N, kBlock + tileSize);
                        for(int i = iBlock; i < iUpper; ++i)
                        {
                            for(int j = jBlock; j < jUpper; ++j)
                            {
                                float val = fourDimRead(O, b, h, i, j, H, N, d);
                                for(int k = kBlock; k < kUpper; ++k)
                                    val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                                fourDimWrite(O, b, h, i, j, H, N, d, val);
                            }
                        }
                    } 
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            #pragma omp parallel for
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                // 将Q的第i行依次与K的所有行相乘，填入ORow
                for(int j = 0; j < N; ++j)
                {
                    float val = 0.0;
                    for(int k = 0; k < d; ++k)
                        val += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    ORow[j] = val;
                }

                // softmax
                float lx = 0.0;
                for(int j = 0; j < N; ++j)
                {
                    float val = ORow[j];
                    ORow[j] = std::exp(val);
                    lx += ORow[j];
                }
                for(int j = 0; j < N; ++j)
                    ORow[j] /= lx;

                // ORow * V(N * d)
                for(int j = 0; j < d; ++j)
                {
                    float val = 0.0;
                    for(int k = 0; k < N; ++k)
                        val += ORow[k] * fourDimRead(V, b, h, k, j, H, N, d);
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
	}
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    for(int b = 0; b < B; ++b)
    {
        for(int h = 0; h < H; ++h)
        {
            for(int jBlock = 0; jBlock < N; jBlock += Bc)
            {
                for(int iBlock = 0; iBlock < N; iBlock += Br)
                {
                    // 计算分块矩阵的乘积
                    for(int j = jBlock; j < std::min(N, jBlock + Bc); ++j)
                    {
                        for(int i = iBlock; i < std::min(N, iBlock + Br); ++i)
                        {
                            float qktVal = 0.0;
                            for(int k = 0; k < d; ++k)             
                                qktVal += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                            twoDimWrite(Sij, i - iBlock, j - jBlock, Bc, qktVal);
                            twoDimWrite(Pij, i - iBlock, j - jBlock, Bc, std::exp(qktVal));
                            // 累加新的lij
                            lij[i - iBlock] += twoDimRead(Pij, i - iBlock, j - jBlock, Bc);
                        }
                    }
                    // 计算新l
                    for(int i = 0; i < Br; ++i)
                        lnew[i] = l[i] + lij[i];
                    // 根据旧l缩放原数据，再用新l计算
                    for(int j = 0; j < d; ++j)
                    {
                        for(int i = iBlock; i < std::min(N, iBlock + Br); ++i)
                        {
                            float val = 0.0;
                            for(int k = jBlock; k < std::min(N, jBlock + Bc); ++k)
                                val += twoDimRead(Pij, i - iBlock, k - jBlock, Bc) * fourDimRead(V, b, h, k, j, H, N, d);
                            float val2 = fourDimRead(O, b, h, i, j, H, N, d);
                            val2 = (val2 * l[i] + val) / lnew[i];
                            fourDimWrite(O, b, h, i, j, H, N, d, val2);
                        }
                    }
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
