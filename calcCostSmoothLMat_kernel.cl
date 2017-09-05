inline int getPenalty(
	int a_DeltaLuma, int a_PenaltyType,
	const int a_DeltaLumaThreshold, const int a_RangeWeightEn,
	const float a_SgmPenaltyP0, const float a_SgmPenaltyP1,	const float a_SgmPenaltyP2, const float *a_PenaltyLumaCoefVec)
{
	if (a_DeltaLuma > a_DeltaLumaThreshold)
	{
        if (a_RangeWeightEn){
            float RangeWeightVal = a_PenaltyLumaCoefVec[a_DeltaLuma-a_DeltaLumaThreshold];
            if (a_PenaltyType == 1)
			{
                return (a_SgmPenaltyP1*RangeWeightVal);
            }
            else
			{
                return (a_SgmPenaltyP2*RangeWeightVal);
            }
        }
        else
		{
            return a_SgmPenaltyP0;
		}
	}
	else if(a_PenaltyType == 1)
	{
		return a_SgmPenaltyP1;
	}
	else
	{
		return a_SgmPenaltyP2;
	}
}

inline int calcCostSmoothMatFunc_CL(
	__global int *a_CostSmooth, __global const int *a_CostData, int i, int j, int i_delta, int j_delta,
	int DeltaLuma, int a_OldCostMin,
	const int a_DeltaLumaThreshold, const int a_RangeWeightEn,
	const float a_SgmPenaltyP0, const float a_SgmPenaltyP1,	const float a_SgmPenaltyP2, const float *a_PenaltyLumaCoefVec,
	const int a_ImgWidth, const int a_ImgHeight, const float a_SgmCostRatio, const float a_SgmPenaltyMax,
	const int a_RangeUpScale, const int a_ResUpscaleFactor)
{
	int NewCostMin;
	int CostDataD;
	int CostDeltaD_gt1, CostDeltaD_0, CostDeltaD_m1, CostDeltaD_p1;
	int CostDeltaMin;
	int P2 = getPenalty(DeltaLuma, 2, a_DeltaLumaThreshold, a_RangeWeightEn, a_SgmPenaltyP0, a_SgmPenaltyP1, a_SgmPenaltyP2, a_PenaltyLumaCoefVec);
	int P1 = getPenalty(DeltaLuma, 1, a_DeltaLumaThreshold, a_RangeWeightEn, a_SgmPenaltyP0, a_SgmPenaltyP1, a_SgmPenaltyP2, a_PenaltyLumaCoefVec);
	int i_delta_idx = i+i_delta;
	int j_delta_idx = j+j_delta;
	int aTmpIndex[4] = {};
	int ImgSize = a_ImgHeight*a_ImgWidth;

	int Limit = j*a_ResUpscaleFactor;
	
	NewCostMin = a_SgmCostRatio * a_CostData[i*a_ImgWidth + j] + a_SgmPenaltyMax; // upper bound
	for (int d = 0; (d < a_RangeUpScale) && (d <= Limit); d++)
	{
		int TmpPosRow = i_delta_idx*a_ImgWidth;
		int TmpPosLayer = d*ImgSize;
		
		aTmpIndex[0] = TmpPosLayer + TmpPosRow + j_delta_idx;			// d*ImgSize + i_delta_idx*a_ImgWidth + j_delta_idx
		aTmpIndex[1] = (TmpPosLayer-ImgSize) + TmpPosRow + j_delta_idx;	// (d-1)*ImgSize + i_delta_idx*a_ImgWidth + j_delta_idx
		aTmpIndex[2] = (TmpPosLayer+ImgSize) + TmpPosRow + j_delta_idx;	// (d+1)*ImgSize + i_delta_idx*a_ImgWidth + j_delta_idx
		aTmpIndex[3] = TmpPosLayer + i*a_ImgWidth + j;					// d*ImgSize + i*a_ImgWidth + j

		int DMij = a_CostData[aTmpIndex[3]];
		CostDataD = a_SgmCostRatio * DMij;
		CostDeltaD_gt1 = a_OldCostMin + P2;
		CostDeltaMin = CostDeltaD_gt1;
		//delta d = 0
		if (d <= j*a_ResUpscaleFactor - 1)
		{
			CostDeltaD_0 = a_CostSmooth[aTmpIndex[0]];
			CostDeltaMin = min(CostDeltaMin, CostDeltaD_0);
		}
		if (d - 1 >= 0)
		{
			// CostDeltaD_m1 = C(p,d) + Lr(p-r,d-1) + P1
			CostDeltaD_m1 = a_CostSmooth[aTmpIndex[1]] + P1;
			CostDeltaMin = min(CostDeltaMin, CostDeltaD_m1);
		}

		if ((d + 1 <= j*a_ResUpscaleFactor - 1) && (d + 1 <a_RangeUpScale))
		{
			CostDeltaD_p1 = a_CostSmooth[aTmpIndex[2]] + P1;
			CostDeltaMin = min(CostDeltaMin, CostDeltaD_p1);
		}

		int NewCost = CostDataD + CostDeltaMin - a_OldCostMin;
		a_CostSmooth[aTmpIndex[3]] = NewCost;

		NewCostMin = min(NewCostMin, NewCost);

	}
	
	return NewCostMin;

}

__kernel void calcCostSmoothLMat_kernel(
	const int a_StartPos, const int a_RangeUpScale, const int a_ResUpscaleFactor, const int a_ImgWidth,
	const int a_ImgHeight, const int a_DeltaLumaThreshold, const int a_RangeWeightEn, const float a_SgmPenaltyP0, 
	const float a_SgmPenaltyP1,	const float a_SgmPenaltyP2, const float a_SgmCostRatio, const float a_SgmPenaltyMax,
	__constant const float *a_PenaltyLumaCoefVec, __global const unsigned char *a_Luma,
	__global const int *a_CostData, __global int *a_CostSmooth)
{
	// Get the work index of the current element to be processed
    int y = get_global_id(0);
	
	int OldCostMin;
	int TmpIndex;
	
	float aLocalCoe[256];
	for (int i = 0 ; i < 256 ; i++)
	{
		aLocalCoe[i] = a_PenaltyLumaCoefVec[i];
	}
	int DeltaLumaThreshold = a_DeltaLumaThreshold;
	int RangeWeightEn = a_RangeWeightEn;
	float SgmPenaltyP0 = a_SgmPenaltyP0;
	float SgmPenaltyP1 = a_SgmPenaltyP1;
	float SgmPenaltyP2 = a_SgmPenaltyP2;
	int ImgWidth = a_ImgWidth;
	int ImgHeight = a_ImgHeight;
	float SgmCostRatio = a_SgmCostRatio;
	float SgmPenaltyMax = a_SgmPenaltyMax;
	int RangeUpScale = a_RangeUpScale;
	int ResUpscaleFactor = a_ResUpscaleFactor;
		
	// Cal Img size
	int ImgSize = a_ImgWidth*a_ImgHeight;

	// Cal Data size
	int DataSize = a_RangeUpScale*ImgSize;
	
	
	int NewCostMin = a_CostData[y*a_ImgWidth + a_StartPos];
	int ImageIndex = y*a_ImgWidth + a_StartPos;
	int TmpIndexCnt = 0;
	for (int d = 0; (d < a_RangeUpScale); d++)
	{
		TmpIndex = TmpIndexCnt + ImageIndex;
		int CostDataAtD = a_CostData[TmpIndex];
		a_CostSmooth[TmpIndex] = CostDataAtD;

		// find minimal cost
		NewCostMin = min(NewCostMin, CostDataAtD);
		
		TmpIndexCnt = TmpIndexCnt + ImgSize;
	}
	OldCostMin= NewCostMin;
	
	// Do the operation
	int TmpIndexRow = y*a_ImgWidth;
	for (int x = a_ImgWidth - 2; x >= 0; x--)
	{
		TmpIndex = TmpIndexRow + x;
		int DeltaLuma = abs(a_Luma[TmpIndex] - a_Luma[TmpIndex + 1]);
		
		OldCostMin = calcCostSmoothMatFunc_CL(
		a_CostSmooth, a_CostData, y, x, 0, 1, DeltaLuma, OldCostMin, DeltaLumaThreshold, RangeWeightEn,
		SgmPenaltyP0, SgmPenaltyP1,	SgmPenaltyP2, aLocalCoe,
		ImgWidth, ImgHeight, SgmCostRatio, SgmPenaltyMax, RangeUpScale, ResUpscaleFactor);
	}
}