/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>

#include <layer_kernels.cuh>

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    cutilCheckMsg("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    cutilCheckMsg("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxProbs;
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    } else {
        kSoftmaxGrad<true><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    }
    cutilCheckMsg("computeSoftmaxGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregSoftmaxGrad: Kernel execution failed");
}



/*
 * E = -P_{ij}*O_{ij} + log(1 + e^{O_{ij}})
 * scores:          (1, numCases)
 * votes:           (1, numCases)
 * entropyCost:     (1, numCases)
 * correctProbs:    (1, numCases)
 * 
 * target:          (1, numCases)
 */
__global__ void kCrossEntropyCost(float* scores, float* votes,  float * entropyCost, float* correctProbs, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

	// if tx is odd like 5
	// the pair is  (4, 5)
	// 
	// if tx is even like 4
	// the pair is  (4, 5)
    if (tx < numCases) {
        float vote1 , vote2;
		float score1, score2;
		if(tx & 0x1){
			vote1 = votes[tx^0x1];
			vote2 = votes[tx];
			score1 =  scores[tx^0x1];
			score2 =  scores[tx];
		}else{
			vote1 = votes[tx];
			vote2 = votes[tx^0x1];
			score1 =  scores[tx];
			score2 =  scores[tx^0x1];
		}
        
		const float p_ij =  vote1/(vote1 + vote2);
		const float o_ij = score1 - score2;
		entropyCost[tx] = -p_ij*o_ij + __logf(1 + __expf(o_ij)); 
        
        if (o_ij * (vote1 - vote2) > 0) {
            correctProbs[tx] = 1;
        } else if(o_ij * (vote1 - vote2)< 0){
            correctProbs[tx] = 0;
        } else {
			correctProbs[tx] = 0.5;
		}
    }
}


/*
 * E = -P_{ij}*O_{ij} + log(1 + e^{O_{ij}})
 * scores:          (1, numCases)
 * votes:           (1, numCases)
 * entropyCost:     (1, numCases)
 * correctProbs:    (1, numCases)
 * 
 * target:          (1, numCases)
 */
__global__ void kCrossEntropyCostNew(float* scores, float* votes,  float * entropyCost, float* correctProbs, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

	// if tx is odd like 5
	// the pair is  (4, 5)
	// 
	// if tx is even like 4
	// the pair is  (4, 5)
    if (tx < (numCases >> 1)) {
        //float vote1 , vote2;
		//float score1, score2;
		const float vote1 = votes[ (tx << 1) ];
		const float vote2 = votes[ (tx << 1)  + 1];
		const float score1 = scores[ (tx << 1) ];
		const float score2 = scores[ (tx << 1) + 1];
        
		const float p_ij =  vote1/(vote1 + vote2);
		const float o_ij = score1 - score2;
		entropyCost[tx << 1] = -p_ij*o_ij + __logf(1 + __expf(o_ij)); 
		entropyCost[(tx << 1) + 1 ] = entropyCost[tx << 1]; 
        
        if (o_ij * (vote1 - vote2) > 0) {
            correctProbs[tx << 1] = 1;
            correctProbs[(tx << 1) + 1] = 1;
        } else if(o_ij * (vote1 - vote2)< 0){
            correctProbs[tx << 1] = 0;
            correctProbs[(tx << 1) + 1] = 0;
        } else {
			correctProbs[tx<<1] = 0.5;
            correctProbs[(tx << 1) + 1] = 0.5;
		}
    }
}

/*
 * E = -P_{ij}*O_{ij} + log(1+e^{O_{ij}})
 * votes:           (1 ,numCases)
 * scores:          (1, numCases)
 * crossEntropy:    (1, numCases)
 * 
 * target:          (1, numCases)
 */
void computeCrossEntropyCost(NVMatrix& votes, NVMatrix& scores, NVMatrix& crossEntropy, NVMatrix& correctProbs_out) {
    
	
	int numCases = scores.getNumCols(); 
    int numOut = scores.getNumRows(); 
	assert(numOut == 1);
	
	#ifdef DEBUG_KERNEL:
	fprintf(stdout, "CrossEntropyCostLayer::fpropActs->computeCrossEntropyCost numCases = %d, numOut = %d\n", numCases, numOut);
	#endif
    assert(votes.getNumElements() == numCases);
    assert(!votes.isTrans());
    assert(!scores.isTrans());
    assert(votes.isContiguous());
    assert(scores.isContiguous());
    
	#ifdef DEBUG_KERNEL:
	fprintf(stdout, "CrossEntropyCostLayer::fpropActs->computeCrossEntropyCost assert OK\n");
	#endif
    crossEntropy.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    
	dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kCrossEntropyCost, cudaFuncCachePreferL1);
    kCrossEntropyCost<<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), crossEntropy.getDevData(), correctProbs_out.getDevData(),numCases, numOut);
    //assert(numCases % 2 == 0);
	//dim3 blocks(DIVUP((numCases >> 1), LOGREG_ERR_THREADS_X), 1);
    //cudaFuncSetCacheConfig(kCrossEntropyCostNew, cudaFuncCachePreferL1);
    //kCrossEntropyCostNew<<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), crossEntropy.getDevData(), correctProbs_out.getDevData(),numCases, numOut);
    cutilCheckMsg("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
}






/*
 * E = -P_{ij}*O_{ij} + log(1+e^{O_{ij}})
 * \nabla_{O_{i}} E = - P_{ij} + e^{O_ij}/(1+e^{O_{ij}}) }
 * \nabla_{O_{j}} E = P_{ij} -  e^{O_ij}/(1+e^{O_{ij}}) }  = - \nabla_{O_{i}} E
 * votes:           (1 ,numCases)
 * scores:          (1, numCases)
 * crossEntropy:    (1, numCases)
 * 
 * target:          (1, numCases)
 */
template <bool add>
__global__ void kCrossEntropyGrad(float* scores, float* votes, float* target, const int numCases,const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    
	// the pair is  (4, 5) 
	// the pair is  (4, 5)
    if (tx < numCases) {
		float vote1, vote2, score1, score2;
		if (tx & 0x1) {
			// odd  O_j  P_{ij} -  e^{O_ij}/(1+e^{O_{ij}}) }  = - \nabla_{O_{i}}
			vote1 = votes[tx^0x1];
			vote2 = votes[tx];
			score1 =  scores[tx^0x1];
			score2 =  scores[tx];
			if(add) {
				target[tx] += - gradCoeff*( vote1 / (vote1 + vote2)  - (1 -  1.0/(1 + __expf(score1 - score2))));
			}else{
				target[tx] = - gradCoeff*( vote1 / (vote1 + vote2)  - (1 -  1.0/(1 + __expf(score1 - score2))));
			}
			
		}else{
			vote1 = votes[tx];
			vote2 = votes[tx^0x1];
			score1 =  scores[tx];
			score2 =  scores[tx^0x1];
			if(add) {
				target[tx] +=  gradCoeff*( vote1 / (vote1 + vote2)  - (1 -  1.0/(1 + __expf(score1 - score2))));
			}else{
				target[tx] =  gradCoeff*( vote1 / (vote1 + vote2)  - (1 -  1.0/(1 + __expf(score1 - score2))));
			}

		}
    }
}

/*
 * E = -P_{ij}*O_{ij} + log(1+e^{O_{ij}})
 * \nabla_{O_{i}} E = - P_{ij} + e^{O_ij}/(1+e^{O_{ij}}) }
 * \nabla_{O_{j}} E = P_{ij} -  e^{O_ij}/(1+e^{O_{ij}}) }  = - \nabla_{O_{i}} E
 * votes:           (1 ,numCases)
 * scores:          (1, numCases)
 * crossEntropy:    (1, numCases)
 * 
 * target:          (1, numCases)
 */
template <bool add>
__global__ void kCrossEntropyGradNew(float* scores, float* votes, float* target, const int numCases,const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    
	// the pair is  (4, 5) 
	// the pair is  (4, 5)
    if (tx < (numCases >> 1)) {
		//float vote1, vote2, score1, score2;
		const float vote1 = votes[tx << 1];
		const float vote2 = votes[(tx << 1) + 1];
		const float score1 = scores[ tx << 1];
		const float score2 = scores[ (tx << 1) + 1];
		
		const float gradProduce = gradCoeff*( vote1 / (vote1 + vote2)  - (1 -  1.0/(1 + __expf(score1 - score2))));
		if (add ) {
		    target[tx << 1] +=  gradProduce;	
			target[ (tx << 1) + 1] += -gradProduce;
		}else{
		    target[tx << 1] =  gradProduce;	
			target[ (tx << 1) + 1] = -gradProduce;
		}
    }
}
/*
 * votes :      (1, numCases)
 * scores :     (1, numCases)
 * target :     (1, numCases)
 * 
 */
void computeCrossEntropyGrad(NVMatrix& votes, NVMatrix& scores, NVMatrix& target, bool add, float coeff) {
    int numCases = scores.getLeadingDim(); 
    int numOut = scores.getFollowingDim(); 
    assert(votes.getNumElements() == numCases);
    assert(votes.isContiguous());
    assert(scores.isContiguous());
    assert(target.isContiguous());
    
	assert(!scores.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, 1);
	
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), 1);
    if (!add) {
        target.resize(scores);
        kCrossEntropyGrad<false><<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), target.getDevData(),numCases, numOut, coeff);
    } else {
        kCrossEntropyGrad<true><<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), target.getDevData(),numCases, numOut, coeff);
    }
	
	/*
    dim3 blocks(DIVUP((numCases>>1), LOGREG_GRAD_THREADS_X), 1);
    if (!add) {
        target.resize(scores);
        kCrossEntropyGradNew<false><<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), target.getDevData(),numCases, numOut, coeff);
    } else {
        kCrossEntropyGradNew<true><<<blocks, threads>>>(scores.getDevData(), votes.getDevData(), target.getDevData(),numCases, numOut, coeff);
    }
	*/
    cutilCheckMsg("computeCrossEntropyGrad: Kernel execution failed");
}

