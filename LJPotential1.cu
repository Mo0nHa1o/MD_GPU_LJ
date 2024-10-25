//program description:
//brwonian paritcles dynamics in a box use GPU
/*
kernel code
first get CellList
then get around particle id per particle
third get hybrid list use bit calculate
firth get force
last update position
*/
//use a flag to mask whether cell of idth particle at the edge of box, use -1 0 1, if at right FlagX=1;
//In the end of every function,there is a __syncthreads to pervent error
//采用csv作为保存数据的文件格式

#include <cufftXt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <math.h>
#include <random> // The header for the generators.
#include <iomanip>
//Definitions=======================================================================
// Define the precision of real numbers, could be float/double.
#define real double
#define Pi 3.1415926535897932384626433832795
#define Zero 0
//typedef double2 Complex;
using namespace std;

struct Particle{
	real* xGPu;//save x position in GPU
	real* yGpu;//save y position in GPU
	int* cellX;//save xth cell of nth particle
	int* cellY;//save yth cell of nth particle
	int* cellList;//cell particle id for all particle, as [maxParticlePerCell*id + offsetCl]
	int* cellOffsetsCl;//offset of every cell list to save particle number in this cell 
	int* particleAroundId;//save ids around this on particle, use rd to judge wether is "around"
	int* particleAroundFlagX;//mask whether cell of idth particle at the edge of box
	int* particleAroundFlayY;//mask whether cell of idth particle at the edge of box
	int* offsetNL;//offset of every particle's neighbor list to save neighbor particle id
	int* NeighborList;//neighbor list
	int* NeighborListFlagX;//translate from particleAroundFlagX
	int* NeighborListFlagY;//translate from particleAroundFlagY
	real* fx;//force on the x direction
	real* fy;//force on the y direction
	real* x0ToUpdateHybridList;//save xGpu[id] to judge whether update hybrid list 
	real* y0ToUpdateHybridList;//save yGpu[id] to judge whether update hybrid list
} PT,pt;

struct Parameter{
	real boxX;//box size X
	real boxY;//box size Y
	real cellSizeX;//cell size in the x direction
	real cellSizeY;//cell size in the y direction
	int cellNumX;//num of cell in the x direction
	int cellNumY;//num of cell in the y direction
	real rho;		//密度
	int maxParticlePerCell;//theory maxmum particle number in one cell
	real rd;//deadline distance to get particleAroundId
	int mask0;//use for bit calculate
	int mask1;//use for bit calculate 	
	real miniInstanceBetweenParticle;//theory minimum distance from two particle
	real r0;//balance position
	real epsilon;//coefficient of force
	int kBT;//kB*T
	int gammaValue;//Viscosity coefficien
	real rOutUpdateList;//update hybrid list when any one particle move a distance greater than rOutUpdateList
	int paraticleNum; //粒子数目
    real tStart;
    real tStop;
    real tStep;
} PM;
//Flag use to update hybrid list=================================================================================================
__global__ real updateHybridListFlag=0;

//input data=====================================================================================================
// 定义变量名称列表结构体
typedef struct NameList {
    const char* vName;    //变量名
    //char *vName;    //变量名
    void* vPtr;     //变量指针
    VType vType;    //变量类型
    int vLen, vStatus; //变量长度、状态
}NameList;

//定义变量
real boxX, boxY, cellSizeX, cellSizeY, rho, miniInstanceBetweenParticle, r0, epsilon;
int cellNumX, cellNumY, maxParticlePerCell, mask0, mask1, kBT, gammaValue, particleNum;
//宏定义
#define NameI(x) {#x,&x,N_I,sizeof(x)/sizeof(int)} //定义整型变量名、指针、类型、长度、状态
#define NameR(x) {#x,&x,N_R,sizeof(x)/sizeof(real)}//定义实型变量名、指针、类型、长度、状态

// 变量名称列表
NameList nameList[] = {
    NameR(boxX),
    NameR(boxY),
    NameR(cellSizeX),
    NameR(cellSizeY),
    NameR(rho),
    NameR(miniInstanceBetweenParticle),
    NameR(r0),
    NameR(epsilon),
    NameI(cellNumX),
    NameI(cellNumY),
    NameI(maxParticlePerCell),
    NameI(mask0),
    NameI(mask1),
    NameI(kBT),
    NameI(gammaValue),
    NameI(particleNum),

};
// 定义宏，用于简化代码
#define NP_I ((int*)(nameList[k].vPtr) + j)
#define NP_R ((real*)(nameList[k].vPtr) + j)

//=======================================================================================================
__device__ void getCellList(int id, Particle PT, Parameter PM);
__device__ void getAroundCellParticleId(int id, Particle PT, Parameter PM);
__device__ void getNeighborList(int id, Particle PT, Parameter PM);
__device__ void updateHybridList(int id, Particle PT, Parameter PM);
__device__ void getForce(int id, Particle PT, Parameter PM);
__device__ void updatePosition(int id, Particle PT, Particle PM);
__device__ void checkAndUpdateHybridList(int id, Particle PT, Parameter PM);
__device__ void clearUpdateHybridListFlag(int id);
__global__ void iterate(Particle PT, Parameter PM);
__global__ int checkUpdateHybridList(int id, real xLast, real yLast, real xNow, real yNow, real rOutUpdateList);
void showProgress(real tNow, real tStart, real tStop, clock_t clockNow, clock_t clockStart);
int GetNameList(int argc, char** argv);
void PrintNameList(FILE* fp);
void Init_Coords(int flag, Particle PT, Parameter PM);
void Init_Parameter();
void Init_System(int argc, char** argv);
void MemAlloc();
void MemFree();
void HostUpdataToDevice();
void DeviceUpdataToHost();
void ExpoConf(const std::string& str_t);
//=================================================================================

int main(int argc, char** argv) {
    clock_t clockNow = clock();
    float tNow = PM.tStart;
    Init_System(argc, argv); //初始化系统    
    HostUpdataToDevice();// 上传数据到设备
    //gpu kernel
    for (tNow = PM.tStart;tNow < PM.tStop;tNow+=tStep) {
        iterate<<<1,PM.particleNum>>>(PT, PM,tNow-tStart);
        if (floor(tNow / P.tExpo) > floor((tNow - tStep) / P.tExpo)) {
            DeviceUpdataToHost();//下载数据到主机
            int te = floor(tNow / P.tExpo);
            str_t = to_string(te);
            ExpoConf(str_t);
            showProgress(tNow,PM.tStart,PM.tStop,clockNow,clock());
    }
    MemFree();//释放内存
    return 0;
}

//==========================================================================================================
__device__ void getCellList(int id,Particle PT,Parameter PM){
	int cellX[id]=std::floor(PT.xGpu[id]/PM.cellSizeX);
	int cellY[id]=std::floor(PT.yGpu[id]/PM.cellSizeY);
	int cellId=cellY*PM.cellNumX+cellX;
	int offsetCL=atomicAdd(&PT.cellOffsetsCL[cellId],1);
	if (offsetCL<PM.maxParticlePerCell){
		PT.cellList[cellId*PM.maxParticlePerCell+offsetCL]=id;
	}else{
		printf("wrong");//append cout error later
	}
	__syncthreads();
}

//==========================================================================================================
__device__ void getAroundCellParticleId(int id,Particle PT,Parameter PM){
	int offsetPAI=0;//particleAroundId put particleId in PAI
	int periodicBoundaryFlagX,periodicBoundaryFlagY;
	int cellXAround,cellYAround;
	int cellAroundId;
	for(int x=-1;x<=1;x++){
		for(int y=-1;y<=1;y++){
			//int cellXAround=cellX+x==-1?cellNumX-1:cellX+x==cellNumX?0:cellX+x;//periodic boundary condition
			//int cellYAround=cellY+y==-1?cellNumY-1:cellY+y==cellNumY?0:cellY+y;
			if(PT.cellX[id]+x==-1){
				cellXAround=cellNumX-1;
				periodicBoundaryFlagX=1;
			}else if(PT.cellX[id]+x==cellNumX){
				cellXAround=0;
				periodicBoundaryFlagX=-1;
					
			}else{
				cellXAround=PT.cellX[id]+x;
				periodicBoundaryFlagX=0;
			}
			if(PT.cellY[id]+y==-1){
				cellYAround=cellNumY-1;
				periodicBoundaryFlagY=1;
			}else if(PT.cellY[id]==cellNumY){
				cellYAround=0;
				periodicBoundaryFlagY=-1;
			}else{
				cellYAround=PT.cellY[id]+y;
				periodicBoundaryFlagY=0;
			}

			cellAroundId=cellYAround*PM.cellNumX+cellXAround;
			for(int i=0;i<cellOffsetsCL[cellAroundId]){
				PT.particleAroundId[offsetPAI]=PT.cellList[cellAroundId*PM.maxParticlePerCell+i];
				PT.particleAroundFlagX[offsetPAI]=periodicBoundaryFlagX;
				PT.particleAroundFlagY[offsetPAI]=periodicBoundaryFlagY;
				offsetPAI++;
			}
		}
	}
	__syncthreads();
}

//==========================================================================================================
__device__ void getNeighborList(int id,Particle PT,Parameter PM){
//get neighborList use bit calculate
	int AX=std::floor(xGpu[id]/PM.miniInstanceBetweenParticle);
	int AY=std::floor(yGpu[id]/PM.miniInstanceBetweenParticle);
	int A0=AY+(AX<<11);//make position real to int
	int A1=A0|PM.mask0;
	int iId,BX,BY,B0,B1,B2,A2,B2;
	int offsetNL;
	for(int i=0;i<offsetPAI;i++){
		iId=PT.particleAroundId[i];
		BX=std::floor(PT.xGpu[iId]/PM.miniIntstanceBetweenParticle);
		BY=std::floor(PT.yGpu[iId]/PM.miniInstanceBetweenParticle);
		B0=BY+(AY<<11);
		B1=B0|PM.mask0;
		A2=(A1-B0)&PM.mask1;
		B2=(B1-A0)&PM.mask1;
		if(!(( (A2&B2==0) | (A2&(B2<<1)) | (A2<<1&B2) )= 0)){
			offsetNL=atomicAdd(&PT.OffsetsNL[id],1);
			PT.NeighborList[id*PM.maxParticlePerNeighbor+offsetNL]=iId;
			PT.NeighborListFlagX[id*PM.maxParticlePerNeighbor+offsetNL]=PT.particleAroundFlagX[i];
			PT.NeighborListFlagY[id*PM.maxParticlePerNeighbor+offsetNL]=PT.particleAroundFlagY[i];
		}
	}
	__syncthreads();
}

//==========================================================================================================
__device__ void updateHybridList(int id,Particle PT,Parameter PM){
	PT.x0ToUpdateHybridList[id]=PT.xGpu[id];
	PT.y0ToUpdateHybridList[id]=PT.yGpu[id];
	getCellList(id,PT,PM);
	getAroundCellParticleId(id,PT,PM); 
	getNeighborList(id,PT,PM);
	__syncthreads();
}

//==========================================================================================================
__device__ void getForce(int id,Particle PT,Parameter PM){
	//get force
	real x,y,xi,yi,dx,dy,dr,f12;
	for(int i=0;i<PT.offsetNL[id];i++){
		x=PT.xGpu[id];
		y=PT.yGpu[id];
		xi=PT.xGpu[PT.NeighborList[id*PM.maxParticlePerNeighbor+i]];
		yi=PT.yGpu[PT.NeighborList[id*PM.maxParticlePerNeighbor+i]];
		dx=(x-xi+PT.NeighborListFlagX[i]*PM.boxX);
		dy=(y-yi+PT.NeighborListFlagY[i]*PM.boxY);
		dr=sqrt(dx*dx+dy*dy);
		f12=24*epsilon*pow(r0,6)*(2*pow(r0,6)-pow(dr,6))/pow(dr,14);
		PT.fx[id]+=f12*dx;
		PT.fy[id]+=f12*dy;
	}
	__syncthreads();
}

//==========================================================================================================
__device__ void updatePosition(int id,Particle PT,Particle PM){
	real fT=sqrt(2*kBT*gamma*tStep);
	xGpu[i]+=fmod((PT.fx[id]*tStep+fT*FRx)/PM.gamma+PM.boxX,PM.boxX);
	yGpu[i]+=fmod((PT.fy[id]*tStep+fT*FRy)/PM.gamma+PM.boxY,PM.boxY);
	__syncthreads();
}

//==========================================================================================================
__device__ void checkAndUpdateHybridList(int id,Particle PT,Parameter PM){
	checkUpdateHybridList(id,PT.x0ToUpdateHybridList[i],PT.y0ToUpdateHybridList[i],xGpu[i],yGpu[i]);
	__syncthreads();
	if(updateHybridListFlag){
		updateHybridList(id,PT,PM);
	}
	__syncthreads();
	clearUpdateHybridList(id);
	__syncthreads();//maybe wrong
}
	
//==========================================================================================================
__device__ void clearUpdateHybridListFlag(int id){
	if(id==0) updateHybridListFlag=0;
}

//==========================================================================================================
__global__ void iterate(Particle PT,Parameter PM,int startFlag){
    if (startFlag)updateHybridList(id, PT, PM);
	int id=blockIdx.y*blockDim.y+blockIdx.x//use one dimentional block ,every particle use one block
	checkAndUpdateHybridList(id,PT,PM);
	getForce(id,PT,PM);
	updatePostion(id,PT,PM);
}

//==========================================================================================================
__global__ int checkUpdateHybridList(int id,real xLast,real yLast,real xNow,real yNow,real rOutUpdateList){
	real dr2=(xNow-xLast)*(xNow-xLast)+(yNow-yLast)*(yNow-yLast);
	if(dr2<rOutUpdateList*rOutUpdateList){
		atomicExch(&dateHybridListFlag,1);
		return 1;	
	}else{
		return 0;
	}
}	

//==========================================================================================================
void showProgress(real tNow,real tStart,real tStop,clock_t clockNow,clock_t clockStart){
	std::cout.flush();
	real progress=((tNow-tStart)/(tStop-tStart);
	real tUsed=double(clockNow,clockStart)/CLOCKS_PER_SEC;
	real tUsePerdiction=tUsed/progress;
	printf("%f\%",progress*100);
	printf("   Peridict:%f",tUsePerdiction);
}

//==========================================================================================================
// 获取名称列表的函数
int GetNameList(int argc, char** argv)
{
    int id, j, k, match, ok;
    char buff[80], * token;
    FILE* fp;

    strcpy(buff, argv[0]);
    strcat(buff, ".in");
    if ((fp = fopen(buff, "r")) == 0)return 0;
    for (k = 0; k < sizeof(nameList) / sizeof(NameList);k++)
        nameList[k].vStatus = 0;
    ok = 1;
    while (1) {
        fgets(buff, 80, fp);
        if (feof(fp))break;
        token = strtok(buff, " \t\n");
        if (!token)break;
        match = 0;
        for (k = 0; k < sizeof(nameList) / sizeof(NameList);k++) {
            if (strcmp(token, nameList[k].vName) == 0) {
                match = 1;
                if (nameList[k].vStatus == 0) {
                    nameList[k].vStatus = 1;
                    for (j = 0;j < nameList[k].vLen; j++) {
                        token = strtok(NULL, " \t\n");
                        if (token) {
                            switch (nameList[k].vType) {
                            case N_I:
                                *NP_I = atol(token);
                                break;
                            case N_R:
                                *NP_R = atof(token);
                                break;
                            }
                        }
                        else {   //定义报错信息

                            nameList[k].vStatus = 2;
                            ok = 0;
                        }
                    }
                    token = strtok(NULL, ", \t\n");
                    if (token) {
                        nameList[k].vStatus = 3;
                        ok = 0;
                    }
                    break;
                }
                else {
                    nameList[k].vStatus = 4;
                    ok = 0;
                }
            }
        }
        if (!match)ok = 0;

    }
    fclose(fp);
    for (k = 0;k < sizeof(nameList) / sizeof(NameList);k++) {
        if (nameList[k].vStatus != 1)ok = 0;
    }
    return ok;
}

//========================================================================================
// 打印名称列表的函数
void PrintNameList(FILE* fp)
{
    int j, k;

    fprintf(fp, "NameList --data\n");
    for (k = 0; k < sizeof(nameList) / sizeof(NameList);k++) {
        fprintf(fp, "%s\t", nameList[k].vName);
        if (strlen(nameList[k].vName) < 8)fprintf(fp, "\t");
        if (nameList[k].vStatus > 0) {
            for (j = 0;j < nameList[k].vLen;j++) {
                switch (nameList[k].vType) {
                case N_I:
                    fprintf(fp, "%d\t", *NP_I);
                    break;
                case N_R:
                    fprintf(fp, "%#g\t", *NP_R);
                    break;
                }
            }
        }
        switch (nameList[k].vStatus) {
        case 0:
            fprintf(fp, "--no data\n");
            break;
        case 1:
            break;
        case 2:
            fprintf(fp, "--missing data\n");
            break;
        case 3:
            fprintf(fp, "--extra data\n");
            break;
        case 4:
            fprintf(fp, "--multiple defined\n");
            break;
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "---------\n");
}

//========================================================================================
void Init_Coords(int flag, Particle PT, Parameter PM) {
    /*
    flag代表系统的初始化方式，flag=0代表均匀分布，flag=1代表随机分布
    当按照均匀分布时，需给定粒子密度，会同时按照初始粒子数目,初始系统的周期盒大小；
    当按照随机分布时，需给定粒子数目，随机生成粒子坐标
    */
    //初始周期盒长度
    int N = PM.ParaticleNum;
    float rho = PM.rho;
    float L = sqrt(N / rho);
    //考虑正方形盒子
    float xBox = L;
    float yBox = L;
    PM.boxX = xBox;
    PM.boxY = yBox;
    int initUcell = sqrt(N); //初始x,y,方向粒子数目
    if flag == 0{
        float d_lattice = L / sqrt(N); //晶格间距
        //均匀分布 系统以原点为中心
        int n, nx, ny;
        n = 0;
        for (ny = 0;ny < initUcell; ny++) {
            for (nx = 0;nx < initUcell; nx++) {
                PT.xGPU[n] = nx * d_lattice;
                PT.yGPU[n] = ny * d_lattice;
                n++;
            }
        }
    }
    //随机分布 均匀分布的随机数生成器
    else if flag == 1{
        std::default_random_engine e;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        e.seed(time(0));

        for (int n = 0; n < N; n++) {
            PT.xGPU[n] = u(e) * xBox;
            PT.yGPU[n] = u(e) * yBox;
        }
    }
}
//initial system=================================================================================================
void Init_Parameter() {
    PM.boxX = nameList.boxX;
    PM.boxY = nameList.boxY;
    PM.cellNumX = nameList.cellNumX;
    PM.cellNumX = nameList.cellNumX;
    PM.cellSizeX = nameList.cellSizeX;
    PM.cellSizeY = nameList.cellSizeY;
    PM.rho = nameList.rho;
    PM.maxParticlePerCell = nameList.maxParticlePerCell;
    PM.mask0 = nameList.mask0;
    PM.mask1 = nameList.mask1;
    PM.miniInstanceBetweenParticle = nameList.miniInstanceBetweenParticle;
    PM.r0 = nameList.r0;
    PM.epsilon = nameList.epsilon;
    PM.gammaValue = nameList.gammaValue;
    PM.kBT = nameList.kBT;
    PM.paraticleNum = nameList.paraticleNum;
    PM.mask0;
    PM.mask1;
}

void Init_System(int argc, char** argv) {
    GetNameList(argc, argv);//导入数据
    PrintNameList(stdout);//打印检查导入数据
    Init_Parameter();
    Init_Coords(0, PT, PM); //均匀分布
    MemAlloc();//分配内存
}

//mem ============================================================================================================

void MemAlloc() {
    // Allocate particle mem in host memory.
    pt.xGpu = new real[PM.paraticleNum];
    pt.yGpu = new real[PM.paraticleNum];
    pt.cellList = new int[PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell];
    pt.cellOffsetsCl = new int[PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell];
    pt.particleAroundId = new int[9 * PM.paraticleNum];
    pt.particleAroundFlagX = new int[PM.paraticleNum];
    pt.particleAroundFlayY = new int[PM.paraticleNum];
    pt.offsetNL = new int[PM.paraticleNum];
    pt.NeighborList = new int[PM.paraticleNum * PM.maxParticlePerCell];
    pt.NeighborListFlagX = new int[PM.paraticleNum];
    pt.NeighborListFlagY = new int[PM.paraticleNum];
    pt.fx = new real[PM.paraticleNum];
    pt.fy = new real[PM.paraticleNum];
    pt.x0ToUpdateHybridList = new int[PM.paraticleNum];
    pt.y0ToUpdateHybridList = new int[PM.paraticleNum];


    // Allocate memory of fields in device.
    cudaMalloc((void**)&PT.cellX, PM.paraticleNum * sizeof(int));
    cudaMalloc((void**)&PT.cellY, PM.paraticleNum * sizeof(int));
    cudaMalloc((void**)&PT.cellList, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(real));
    cudaMalloc((void**)&PT.cellOffsetsCl, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(real));
    cudaMalloc((void**)&PT.particleAroundId, 9 * PM.paraticleNum * sizeof(int));
    cudaMalloc((void**)&PT.particleAroundFlagX, PM.paraticleNum);
    cudaMalloc((void**)&PT.particleAroundFlayY, PM.paraticleNum);
    cudaMalloc((void**)&PT.offsetNL, PM.paraticleNum);
    cudaMalloc((void**)&PT.NeighborList, PM.paraticleNum * PM.maxParticlePerCell * sizeof(int));
    cudaMalloc((void**)&PT.NeighborListFlagX, PM.paraticleNum * sizeof(int));
    cudaMalloc((void**)&PT.NeighborListFlagY, PM.paraticleNum * sizeof(int));
    cudaMalloc((void**)&PT.fx, PM.paraticleNum * sizeof(real));
    cudaMalloc((void**)&PT.fy, PM.paraticleNum * sizeof(real));
    cudaMalloc((void**)&PT.x0ToUpdateHybridList, PM.paraticleNum * sizeof(real));
    cudaMalloc((void**)&PT.y0ToUpdateHybridList, PM.paraticleNum * sizeof(real));
    cudaMalloc((void**)&PT.xGpu, PM.paraticleNum * sizeof(real));
    cudaMalloc((void**)&PT.yGpu, PM.paraticleNum * sizeof(real));
}
//===========================================================================
void MemFree() {
    // Free host memory
    delete[] pt.xGpu;
    delete[] pt.yGpu;
    delete[] pt.cellList;
    delete[] pt.cellOffsetsCl;
    delete[] pt.particleAroundId;
    delete[] pt.particleAroundFlagX;
    delete[] pt.particleAroundFlayY;
    delete[] pt.offsetNL;
    delete[] pt.NeighborList;
    delete[] pt.NeighborListFlagX;
    delete[] pt.NeighborListFlagY;
    delete[] pt.fx;
    delete[] pt.fy;
    delete[] pt.x0ToUpdateHybridList;
    delete[] pt.y0ToUpdateHybridList;

    // Free device memory
    cudaFree(PT.xGpu);
    cudaFree(PT.yGpu);
    cudaFree(PT.cellX);
    cudaFree(PT.cellY);
    cudaFree(PT.cellList);
    cudaFree(PT.cellOffsetsCl);
    cudaFree(PT.particleAroundId);
    cudaFree(PT.particleAroundFlagX);
    cudaFree(PT.particleAroundFlayY);
    cudaFree(PT.offsetNL);
    cudaFree(PT.NeighborList);
    cudaFree(PT.NeighborListFlagX);
    cudaFree(PT.NeighborListFlagY);
    cudaFree(PT.fx);
    cudaFree(PT.fy);
    cudaFree(PT.x0ToUpdateHybridList);
    cudaFree(PT.y0ToUpdateHybridList);
}
//上传=============================================================================================
void HostUpdataToDevice() {
    cudaMemcpy(pt.xGpu, PT.xGPU, PM.paraticleNum * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(pt.xGpu, PT.xGPU, PM.paraticleNum * sizeof(real), cudaMemcpyHostToDevice);
}
//下载=============================================================================================
void DeviceUpdataToHost() {
    cudaMemcpy(PT.xGPU, pt.xGpu, PM.paraticleNum * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(PT.yGPU, pt.yGpu, PM.paraticleNum * sizeof(real), cudaMemcpyDeviceToHost);
}
//output===========================================================================================

void ExpoConf(const std::string& str_t) {
    std::ofstream ConfFile;
    //设置输出精度
    int PrecData = 8;

    // 文件名
    std::string ConfFileName = "data/conf_" + str_t + ".dat";
    ConfFile.open(ConfFileName.c_str());

    if (!ConfFile.is_open()) {
        std::cerr << "无法打开文件: " << ConfFileName << std::endl;
        return;
    }
    for (int idx = 0; idx < PM.paraticleNum; idx++) {
        // 使用固定格式和精度输出数据
        ConfFile << std::fixed << std::setprecision(PrecData)
            << PT.xGpu[idx] << ' '
            << PT.yGpu[idx];
        ConfFile << std::endl; // 换行
    }

    ConfFile.close();
}


