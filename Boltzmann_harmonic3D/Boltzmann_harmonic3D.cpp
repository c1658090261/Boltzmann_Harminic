#include<iostream>
#include<cmath>
#include<cstdlib>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<string>
#define NP 50000000
////////paralleled solver of LBM for high-order (k=2) nonlinear polyharmonic equations/////
using namespace std;
const int Q=13  ;//discrete velocities
const int Np=100;//number of grid points per unit characteristic length
const int NX=Np;  //spatial discretization in x direction
const int NY=Np;  //spatial discretization in y direction
const int NZ=Np;  //spatial discretization in y direction
const double pi=3.1415926;//constant 

double U[NX+1][NY+1][NZ+1];//solution of the order-reduced polyharmonic equation obtained by using LBM solver(fucntion phsi(x,y,z) in the paper)
double V[NX+1][NY+1][NZ+1];//analytic solution of the polyharmonic equation
double U1[NX+1][NY+1][NZ+1];//solution of the harmonic equation obtained by using LBM solver (final reduced, fucntion h(x,y,z) in the paper)
double U0[NX+1][NY+1][NZ+1];//solution of the previous iteration step, used to calculate relative error for each iteration step
double f[NX+1][NY+1][NZ+1][Q];//distribution fucntion before evolution (streaming and collision)
double F[NX+1][NY+1][NZ+1][Q];//distribution fucntion after collision
double M[NX+1][NY+1][NZ+1][Q];//distribution fucntion after streaming
double feq[NX+1][NY+1][NZ+1][Q];//equilibrium distribution fucntion 
double g[NX+1][NY+1][NZ+1];//source term of the target PDEs (polyharmonic equation)
int e[Q][3]={{0,0,0},{1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},{-1,-1,0},{-1,1,0},{-1,0,-1},{-1,0,1},{0,-1,-1},{0,-1,1}};//unit discrete velocities
float w[Q]={1.0/2,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24};//weight coefficients for each duscrete direction
float w_bar[Q]={0,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12};//weight coefficients for evolution equation, see Ref.37 in the paper
int i,j,z,k,ip,jp,zp,n,NN;
double source,Lamda,tau_f,error,alpha,niu,r,RSME;//variables of the numerical algorithm
double R[100000];//mean square root error for each evolution step
void init();//initial procedure for the order-reduced polyharmonic equation
void init1();//initial procedure for the harmonic equation 
void evolution();//evolution procedure for the order-reduced polyharmonic equation
void evolution1();//evolution procedure for the harmonic equation (final reduced)
void output_x1(int m);//output a slice on y-z plane at x=0.2
void output_y1(int m);//output a slice on x-z plane at y=0.2
void output_z1(int m);//output a slice on x-y plane at z=0.2
void output_x2(int m);//output a slice on y-z plane at x=0.8
void output_y2(int m);//output a slice on x-z plane at y=0.8
void output_z2(int m);//output a slice on x-y plane at z=0.8
void Error();//relative error for the order-reduced polyharmonic equation
void Error1();//relative error for harmonic equation (final reduced)
void output_error();//output mean square root error
void output_subtractionx1();//output a slice of subtraction between nuemrical results and analytic solution on y-z plane at x=0.2
void output_subtractionx2();//output a slice of subtraction between nuemrical results and analytic solution on x-z plane at y=0.2
void output_subtractiony1();//output a slice of subtraction between nuemrical results and analytic solution on x-y plane at z=0.2
void output_subtractiony2();//output a slice of subtraction between nuemrical results and analytic solution on y-z plane at x=0.8
void output_subtractionz1();//output a slice of subtraction between nuemrical results and analytic solution on x-z plane at y=0.8
void output_subtractionz2();//output a slice of subtraction between nuemrical results and analytic solution on x-y plane at z=0.8
void output_error()
{
	ostringstream name;
	name<<"RMSE.dat";
	ofstream fout(name.str().c_str());
	for(i=0;i<=NN;i++)
	{
		fout<<i<<"      "<<R[i]<<endl;
	}
}
void output_subtractionx1() 
{
	ostringstream name;
	name<<"subtraction_x=0.2.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<endl;
	
	for(j=0;j<=NY;j++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1[20][j][z]-V[20][j][z]<<endl;
		}
	}
}
void output_subtractionx2() 
{
	ostringstream name;
	name<<"subtraction_x=0.8.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<endl;
	
	for(j=0;j<=NY;j++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1[80][j][z]-V[80][j][z]<<endl;
		}
	}
}
void output_subtractiony1()
{
	ostringstream name;
	name<<"subtraction_y=0.2.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1[i][20][z]-V[i][20][z]<<endl;
		}
	}
}
void output_subtractiony2()
{
	ostringstream name;
	name<<"subtraction_y=0.8.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1[i][80][z]-V[i][80][z]<<endl;
		}
	}
}
void output_subtractionz1() 
{
	ostringstream name;
	name<<"subtraction_z=0.2.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",J="<<Np+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1[i][j][20]-V[i][j][20]<<endl;
		}
	}
}
void output_subtractionz2()
{
	ostringstream name;
	name<<"subtraction_z=0.8.dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",J="<<Np+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1[i][j][80]-V[i][j][80]<<endl;
		}
	}
}
int main()
{
	using namespace std;

	for(i=0;i<=NX;i++)//initiation for the order-reduced polyharmonic equation                                         
	{                                                                                             
		for(j=0;j<=NY;j++)                                                                        
		{                                                                                         
			for(z=0;z<=NZ;z++)                                                                   
			{                                                                                    
				V[i][j][z]=cos(pi*double(i)/Np)*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);       
			}                                                                                   
		}                                                                                       
	}                                                                                             
	init();//initiation for the order-reduced polyharmonic equation                               
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////repeat this procedure if k is larger than 2                                                       /////
//////but notice the source term g should be replaced by U for each reduction of order                  /////
	for(n=0;;n++)	                                                                                    /////
	{                                                                                                   /////
		NN=n;                                                                                           /////
		evolution();//solving the order-reduced polyharmonic equation                                   /////
		if(n%100==0)                                                                                    /////
		{                                                                                               /////
			Error();                                                                                    /////
			cout<<"The"<<n<<"th computation result:"<<endl;                                             /////
			cout<<"The max absolute error is:"<<setiosflags(ios::scientific)<<error<<endl;              /////
			if(error<1.0e-15)//convergence condition                                                    /////
			{                                                                                           /////
				break;                                                                                  /////
			}                                                                                           /////
		}                                                                                               /////
	}                                                                                                   /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	init1();//initiation for the harmonic equation (finally reduced)
	for(n=0;;n++)	
	{
		NN=n;
		evolution1();//solving the harmonic equation (finally reduced)
		RSME=0;	
		for(i=0;i<=NX;i++)//calculate mean square root error
		{
			for(j=0;j<=NY;j++)
			{
				for(z=0;z<=NZ;z++)
				{
					RSME+=(U1[i][j][z]-V[i][j][z])*(U1[i][j][z]-V[i][j][z]);
				}
			}
		}
		R[NN]=sqrt(RSME/(Np*Np));		
		if(n%100==0)
		{
			Error1();
			cout<<"The"<<n<<"th computation result:"<<endl;
			cout<<"The max absolute error is:"<<setiosflags(ios::scientific)<<error<<endl;
			if(error<1.0e-15)//convergence condition 
			{
				output_x1(n);
				output_y1(n);
				output_z1(n);
				output_x2(n);
				output_y2(n);
				output_z2(n);
				output_subtractionx1();
				output_subtractionx2();
				output_subtractiony1();
				output_subtractiony2();
				output_subtractionz1();
				output_subtractionz2();
				output_error();
				break;
			}
		}
	}
	return 0;	
}
void init()
{
	source=1.0;//if source is 0 it is the polyharmonic equation without source term.
	Lamda=1.0/3.0;//constant for building equilibrium distribution fucntion
	tau_f=1.0;//single relaxation time
	std::cout<<"tau_f="<<tau_f<<endl;
	std::cout<<"Lamda="<<Lamda<<endl;

	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				U[i][j][z]=0;//initiating the entire field with 0s
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U[i][j][z] ;//calculating equilirum disribution fucntion
						f[i][j][z][k]=feq[i][j][z][k];//let equilirum disribution fucntion be the distribution fucntion for the initial step
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}	
}
void init1()
{
	source=1.0;
	Lamda=1.0/3.0;
	tau_f=1.0;
	std::cout<<"tau_f="<<tau_f<<endl;
	std::cout<<"Lamda="<<Lamda<<endl;

	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				U1[i][j][z]=0;
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U1[i][j][z] ;
						f[i][j][z][k]=feq[i][j][z][k];
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U1[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}	
}
void evolution()
{
	for(i=0;i<=NX;i++)//calculating source term and equlibirum distribution fucntion for the preparation of evolution
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				g[i][j][z]=source*(9.0*pi*pi*pi*pi*cos(pi*double(i)/Np)*sin(pi*double(j)/Np)*sin(pi*double(z)/Np));
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U[i][j][z] ;
						f[i][j][z][k]=feq[i][j][z][k];
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}
	for(i=0;i<=NX;i++)//collision
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					M[i][j][z][k]=f[i][j][z][k]+(feq[i][j][z][k]-f[i][j][z][k])/tau_f+w_bar[k]*(0.5-tau_f)*g[i][j][z]*Lamda/(Np*Np);
				}
			}
		}
	}
	for(i=1;i<NX;i++)//streaming
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					ip=i-e[k][0];
					jp=j-e[k][1];
					zp=z-e[k][2];
					F[i][j][z][k]=M[ip][jp][zp][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//calculation of Marcroscopic variables(solutions of the target PDEs)
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				U0[i][j][z]=U[i][j][z];
				U[i][j][z]=0;
				for(k=0;k<Q;k++)
				{
					f[i][j][z][k]=F[i][j][z][k];
					U[i][j][z]+=f[i][j][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//calculation of feq ready for the boundary conditions
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}
////////////////////////////boundary conditions////////////////////////
	for(j=0;j<=NY;j++)//left and right walls
	{
		for(z=0;z<=NZ;z++)
		{
			U[0][j][z]=-3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
			U[NX][j][z]=3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
		//	U[0][j][z]=U[1][j][z];    //Neumann boundary conditions
		//	U[NX][j][z]=U[NX-1][j][z];//Neumann boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[0][j][z][k]=(1.0-w[k])*U[0][j][z];
					feq[NX][j][z][k]=(1.0-w[k])*U[NX][j][z];
					f[0][j][z][k]=feq[0][j][z][k]+f[1][j][z][k]-feq[1][j][z][k];
					f[NX][j][z][k]=feq[NX][j][z][k]+f[NX-1][j][z][k]-feq[NX-1][j][z][k];
				}
				else 
				{
					feq[0][j][z][k]=w[k]*U[0][j][z];
					feq[NX][j][z][k]=w[k]*U[NX][j][z];
					f[0][j][z][k]=feq[0][j][z][k]+f[1][j][z][k]-feq[1][j][z][k];
					f[NX][j][z][k]=feq[NX][j][z][k]+f[NX-1][j][z][k]-feq[NX-1][j][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//front and back walls
	{
		for(z=0;z<=NZ;z++)
		{
			U[i][NY][z]=0;//Dirichlet boundary conditions
			U[i][0][z]=0;//Dirichlet boundary conditions
		//	U[i][NY][z]=U[i][NY-1][z];//Neumann boundary conditions
		//	U[i][0][z]=U[i][1][z];//Neumann boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[i][0][z][k]=(1.0-w[k])*U[i][0][z];
					feq[i][NY][z][k]=(1.0-w[k])*U[i][NY][z];
					f[i][0][z][k]=feq[i][0][z][k]+f[i][1][z][k]-feq[i][1][z][k];
					f[i][NY][z][k]=feq[i][NY][z][k]+f[i][NY-1][z][k]-feq[i][NY-1][z][k];
				}
				else
				{
					feq[i][0][z][k]=w[k]*U[i][0][z];
					feq[i][NY][z][k]=w[k]*U[i][NY][z];
					f[i][0][z][k]=feq[i][0][z][k]+f[i][1][z][k]-feq[i][1][z][k];
					f[i][NY][z][k]=feq[i][NY][z][k]+f[i][NY-1][z][k]-feq[i][NY-1][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//up and down walls
	{
		for(j=1;j<NY;j++)
		{
			U[i][j][0]=0;//Dirichlet boundary conditions
			U[i][j][NZ]=0;//Dirichlet boundary conditions
		//	U[i][j][0]=U[i][j][0];//Neumann boundary conditions
		//	U[i][j][NZ]=U[i][j][NZ-1];//Neumann boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[i][j][0][k]=(1.0-w[k])*U[i][j][0];
					feq[i][j][NZ][k]=(1.0-w[k])*U[i][j][NZ];
					f[i][j][0][k]=feq[i][j][0][k]+f[i][j][1][k]-feq[i][j][1][k];
					f[i][j][NZ][k]=feq[i][j][NZ][k]+f[i][j][NZ-1][k]-feq[i][j][NZ-1][k];
				}
				else 
				{
					feq[i][j][0][k]=w[k]*U[i][j][0];
					feq[i][j][NZ][k]=w[k]*U[i][j][NZ];
					f[i][j][0][k]=feq[i][j][0][k]+f[i][j][1][k]-feq[i][j][1][k];
					f[i][j][NZ][k]=feq[i][j][NZ][k]+f[i][j][NZ-1][k]-feq[i][j][NZ-1][k];
				}
			}
		}
	}
////////////////////////////done////////////////////////////////
}
void evolution1()
{
	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U1[i][j][z] ;
						f[i][j][z][k]=feq[i][j][z][k];
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U1[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}
	for(i=0;i<=NX;i++)   //collision
	{
		for(j=0;j<=NY;j++)
		{
			for(z=0;z<=NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					M[i][j][z][k]=f[i][j][z][k]+(feq[i][j][z][k]-f[i][j][z][k])/tau_f+w_bar[k]*(0.5-tau_f)*U[i][j][z]*Lamda/(Np*Np);
				}
			}
		}
	}
	for(i=1;i<NX;i++)   //streaming
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					ip=i-e[k][0];
					jp=j-e[k][1];
					zp=z-e[k][2];
					F[i][j][z][k]=M[ip][jp][zp][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)   //calculation of Marcroscopic variables
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				U0[i][j][z]=U1[i][j][z];
				U1[i][j][z]=0;
				for(k=0;k<Q;k++)
				{
					f[i][j][z][k]=F[i][j][z][k];
					U1[i][j][z]+=f[i][j][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//calculation of feq ready for the boundary conditions
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq[i][j][z][k]=(1.0-w[k])*U1[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
					else 
					{
						feq[i][j][z][k]=w[k]*U1[i][j][z];
						f[i][j][z][k]=feq[i][j][z][k];
					}
				}
			}
		}
	}
////////////////////////////boundary conditions////////////////////////
	for(j=0;j<=NY;j++)//left and right walls
	{
		for(z=0;z<=NZ;z++)
		{
			U1[0][j][z]=sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
			U1[NX][j][z]=-sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[0][j][z][k]=(1.0-w[k])*U1[0][j][z];
					feq[NX][j][z][k]=(1.0-w[k])*U1[NX][j][z];
					f[0][j][z][k]=feq[0][j][z][k]+f[1][j][z][k]-feq[1][j][z][k];
					f[NX][j][z][k]=feq[NX][j][z][k]+f[NX-1][j][z][k]-feq[NX-1][j][z][k];
				}
				else 
				{
					feq[0][j][z][k]=w[k]*U1[0][j][z];
					feq[NX][j][z][k]=w[k]*U1[NX][j][z];
					f[0][j][z][k]=feq[0][j][z][k]+f[1][j][z][k]-feq[1][j][z][k];
					f[NX][j][z][k]=feq[NX][j][z][k]+f[NX-1][j][z][k]-feq[NX-1][j][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//front and back
	{
		for(z=0;z<=NZ;z++)
		{
			U1[i][NY][z]=0;//Dirichlet boundary conditions
			U1[i][0][z]=0;//Dirichlet boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[i][0][z][k]=(1.0-w[k])*U1[i][0][z];
					feq[i][NY][z][k]=(1.0-w[k])*U1[i][NY][z];
					f[i][0][z][k]=feq[i][0][z][k]+f[i][1][z][k]-feq[i][1][z][k];
					f[i][NY][z][k]=feq[i][NY][z][k]+f[i][NY-1][z][k]-feq[i][NY-1][z][k];
				}
				else
				{
					feq[i][0][z][k]=w[k]*U1[i][0][z];
					feq[i][NY][z][k]=w[k]*U1[i][NY][z];
					f[i][0][z][k]=feq[i][0][z][k]+f[i][1][z][k]-feq[i][1][z][k];
					f[i][NY][z][k]=feq[i][NY][z][k]+f[i][NY-1][z][k]-feq[i][NY-1][z][k];
				}
			}
		}
	}
	for(i=1;i<NX;i++)//up and down
	{
		for(j=1;j<NY;j++)
		{
			U1[i][j][0]=0;//Dirichlet boundary conditions
			U1[i][j][NZ]=0;//Dirichlet boundary conditions
			for(k=0;k<Q;k++)
			{
				if(k==0)
				{
					feq[i][j][0][k]=(1.0-w[k])*U1[i][j][0];
					feq[i][j][NZ][k]=(1.0-w[k])*U1[i][j][NZ];
					f[i][j][0][k]=feq[i][j][0][k]+f[i][j][1][k]-feq[i][j][1][k];
					f[i][j][NZ][k]=feq[i][j][NZ][k]+f[i][j][NZ-1][k]-feq[i][j][NZ-1][k];
				}
				else 
				{
					feq[i][j][0][k]=w[k]*U1[i][j][0];
					feq[i][j][NZ][k]=w[k]*U1[i][j][NZ];
					f[i][j][0][k]=feq[i][j][0][k]+f[i][j][1][k]-feq[i][j][1][k];
					f[i][j][NZ][k]=feq[i][j][NZ][k]+f[i][j][NZ-1][k]-feq[i][j][NZ-1][k];
				}
			}
		}
	}
////////////////////////////done////////////////////////////////
}
void output_x1(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_yz_x=0.2-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<NX+1<<",K="<<NY+1<<",F=POINT"<<endl;
	
	for(j=0;j<=NY;j++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1[20][j][z]<<endl;
		}
	}
}
void output_x2(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_yz_x=0.8-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<NX+1<<",K="<<NY+1<<",F=POINT"<<endl;
	
	for(j=0;j<=NY;j++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1[80][j][z]<<endl;
		}
	}
}
void output_y1(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_xz_y=0.2-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<NX+1<<",K="<<NZ+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1[i][20][z]<<endl;
		}
	}
}
void output_y2(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_xz_y=0.8-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<NX+1<<",K="<<NZ+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(z=0;z<=NZ;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1[i][80][z]<<endl;
		}
	}
}
void output_z1(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_xy_z=0.2-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<NX+1<<",J="<<NY+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1[i][j][20]<<endl;
		}
	}
}
void output_z2(int m) // output
{
	ostringstream name;
	name<<"3D-Laplace_xy_z=0.8-plane_"<<m<<".dat";
	ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<NX+1<<",J="<<NY+1<<",F=POINT"<<endl;
	
	for(i=0;i<=NX;i++)
	{
		for(j=0;j<=NY;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1[i][j][80]<<endl;
		}
	}
}
void Error()
{
	double temp1,temp2;
	temp1=0;
	temp2=0;
	for(i=1;i<NX;i++)
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				temp1+=(U[i][j][z]-U0[i][j][z])*(U[i][j][z]-U0[i][j][z]);
				temp2+=(U[i][j][z]*U[i][j][z]);
			}
		}
	}
	temp1=sqrt(temp1);
	temp2=sqrt(temp2);
	error=temp1/(temp2+1e-30);
}
void Error1()
{
	double temp1,temp2;
	temp1=0;
	temp2=0;
	for(i=1;i<NX;i++)
	{
		for(j=1;j<NY;j++)
		{
			for(z=1;z<NZ;z++)
			{
				temp1+=(U1[i][j][z]-U0[i][j][z])*(U1[i][j][z]-U0[i][j][z]);
				temp2+=(U1[i][j][z]*U1[i][j][z]);
			}
		}
	}
	temp1=sqrt(temp1);
	temp2=sqrt(temp2);
	error=temp1/(temp2+1e-30);
}

