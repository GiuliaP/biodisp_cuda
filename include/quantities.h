#ifndef	QUANTITIES_H
#define QUANTITIES_H

#define pigreco 3.1416//5926535897932384626433832795
#define EPS_float 0.000001

#define maxLengthKernel 32
#define nFilters1D 9
#define taps 11
#define nOrient 8
#define nPhase 7
#define angularBandwidth (pigreco / (float)nOrient)
#define centralPhaseIdx ((nPhase-1)/2)

#define f0 0.25
#define nFilter 8
#define thEnergy 0.00000001
#define thOri 0.0
#define thcalcCenterOfMass 0.7
#define nMinOri 4
#define nScale 4

#define filtGaussRadius 2
#define filtGaussLength (filtGaussRadius*2 + 1)

#define nTempMatricesCV (nEye*nCoupleOri*nTempAnswers)
#define nTempMatricesCU (nEye*nCoupleOri*nTempAnswers+nOrient)
#define nTempMatricesCVoneOri (nPhase+2)
#define nTempMatricesCUoneOri (nPhase+2)
#define nTempMatricesCVrepOriRepPhase (2*2)
#define nTempMatricesCUrepOriRepPhase (2*2+1)

#define nStrCV (24+12+1)
#define nStreamCV (8+4)
#define nStrCU (24+12+1)
#define nStreamCU (8+4)

#define nEye 2
#define nResults 2
#define nCoupleOri 3
#define nTempAnswers 6

#define nFunc 6

#endif