/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2018 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */
//  forked from package RcppDist
#include <RcppArmadillo.h>
#include "mvnorm.h"
#include "wishart.h"

#include <ctime>
#include "common.h"
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
//#include "heterbart.h"
#include "rtnorm.h"
#include "rtgamma.h"
#include "lambda.h"

#ifndef NoRcpp

#define MDRAW(a, b) Mdraw(a, b)
#define YDRAW(a, b) Ydraw(a, b)
#define UMDRAW(a, b) uMdraw(a, b)
#define UYDRAW(a, b) uYdraw(a, b)

RcppExport SEXP crBARTmediation(SEXP _typeM,   // 1:continuous, 2:binary, 3:multinomial
                                SEXP _typeY,   // 1:continuous, 2:binary, 3:multinomial
                                SEXP _in,      // number of observations in training data
                                SEXP _ipm,     // dimension of matX
                                SEXP _imatX,   // matX, train, pm x n  // (transposed so rows are contiguous in memory)
                                SEXP _iM,      // M, train, n x 1
                                SEXP _ipy,     // dimension of matM
                                SEXP _imatM,   // matM, train, py x n  // (transposed so rows are contiguous in memory)
                                SEXP _iY,      // Y, train, n x 1
                                SEXP _iu_index,
                                SEXP _in_j_vec,
                                SEXP _iuM,
                                SEXP _iuY,
                                SEXP _iJ,
                                SEXP _iB_uM,
                                SEXP _iB_uY,
                                SEXP _inumtree, // number of trees
                                SEXP _imatXnc,  // number of cut points
                                SEXP _imatMnc,  // number of cut points
                                SEXP _inumdraw, // number of kept draws (except thinnning)
                                SEXP _iburn,    // number of burn-in draws skipped
                                SEXP _ithin,    // thinning
                                SEXP _ipower,
                                SEXP _ibase,
                                SEXP _Moffset,
                                SEXP _Yoffset,
                                SEXP _iMtau,
                                SEXP _iYtau,
                                SEXP _inu,
                                SEXP _iMlambda,
                                SEXP _iYlambda,
                                SEXP _iMsigest,
                                SEXP _iYsigest,
                                SEXP _idart,    // dart prior: true(1)=yes, false(0)=no
                                SEXP _itheta,
                                SEXP _iomega,
                                SEXP _ia,       // param a for sparsity prior
                                SEXP _ib,       // param b for sparsity prior
                                SEXP _imatXrho, // param matXrho for sparsity prior (default=pm)
                                SEXP _imatMrho, // param matMrho for sparsity prior (default=py)
                                SEXP _iaug,     // category strategy:  true(1)=data augment 
                                SEXP _inprintevery,
                                SEXP _matXinfo,
                                SEXP _matMinfo){
  //process args
  int typeM = Rcpp::as<int>(_typeM);
  int typeY = Rcpp::as<int>(_typeY);
  size_t n = Rcpp::as<int>(_in);
  size_t pm = Rcpp::as<int>(_ipm);
  Rcpp::NumericVector matXv(_imatX);
  double *imatX = &matXv[0];
  Rcpp::NumericVector Mv(_iM); 
  double *iM = &Mv[0];
  size_t py = Rcpp::as<int>(_ipy);
  Rcpp::NumericVector matMv(_imatM);
  double *imatM = &matMv[0];
  Rcpp::NumericVector Yv(_iY); 
  double *iY = &Yv[0];
  Rcpp::IntegerVector _u_index(_iu_index);
  int *u_index = &_u_index[0];
  Rcpp::IntegerVector _n_j_vec(_in_j_vec);
  int *n_j_vec = &_n_j_vec[0];
  Rcpp::NumericVector _uM(_iuM);
  double *uM = &_uM[0];
  Rcpp::NumericVector _uY(_iuY);
  double *uY = &_uY[0];
  size_t J = Rcpp::as<int>(_iJ);
  double B_uM = Rcpp::as<double>(_iB_uM);
  double B_uY = Rcpp::as<double>(_iB_uY);
  size_t numtree = Rcpp::as<int>(_inumtree);
  Rcpp::IntegerVector _matXnc(_imatXnc);
  int *matXnc = &_matXnc[0];
  Rcpp::IntegerVector _matMnc(_imatMnc);
  int *matMnc = &_matMnc[0];
  size_t numdraw = Rcpp::as<int>(_inumdraw);
  size_t burn = Rcpp::as<int>(_iburn);
  size_t thin = Rcpp::as<int>(_ithin);
  double mybeta = Rcpp::as<double>(_ipower);
  double alpha = Rcpp::as<double>(_ibase);
  double Moffset = Rcpp::as<double>(_Moffset);
  double Yoffset = Rcpp::as<double>(_Yoffset);
  double Mtau = Rcpp::as<double>(_iMtau);
  double Ytau = Rcpp::as<double>(_iYtau);
  double nu = Rcpp::as<double>(_inu);
  double Mlambda = Rcpp::as<double>(_iMlambda);
  double Ylambda = Rcpp::as<double>(_iYlambda);
  double iMsigest=Rcpp::as<double>(_iMsigest);
  double iYsigest=Rcpp::as<double>(_iYsigest);
  bool dart;
  if(Rcpp::as<int>(_idart)==1) dart=true;
  else dart=false;
  double a = Rcpp::as<double>(_ia);
  double b = Rcpp::as<double>(_ib);
  double matXrho = Rcpp::as<double>(_imatXrho);
  double matMrho = Rcpp::as<double>(_imatMrho);
  bool aug;
  if(Rcpp::as<int>(_iaug)==1) aug=true;
  else aug=false;
  double theta = Rcpp::as<double>(_itheta);
  double omega = Rcpp::as<double>(_iomega);
  size_t nkeeptrain = numdraw/thin;     // Rcpp::as<int>(_inkeeptrain);
  size_t nkeeptreedraws = numdraw/thin; // Rcpp::as<int>(_inkeeptreedraws);
  size_t printevery = Rcpp::as<int>(_inprintevery);
  
  Rcpp::NumericMatrix matXvarprb(nkeeptreedraws,pm);
  Rcpp::NumericMatrix matMvarprb(nkeeptreedraws,py);
  Rcpp::IntegerMatrix matXvarcnt(nkeeptreedraws,pm);
  Rcpp::IntegerMatrix matMvarcnt(nkeeptreedraws,py);
  Rcpp::NumericMatrix matXinfo(_matXinfo);
  Rcpp::NumericMatrix matMinfo(_matMinfo);
  Rcpp::NumericVector Msdraw(nkeeptrain);
  Rcpp::NumericVector Ysdraw(nkeeptrain);
  Rcpp::NumericVector muMudraw(nkeeptrain);
  Rcpp::NumericVector muYudraw(nkeeptrain);
  Rcpp::NumericVector sigMudraw(nkeeptrain);
  Rcpp::NumericVector sigYudraw(nkeeptrain);
  Rcpp::NumericVector rhoMYudraw(nkeeptrain);
  Rcpp::NumericMatrix Mdraw(nkeeptrain,n);
  Rcpp::NumericMatrix Ydraw(nkeeptrain,n);
  Rcpp::NumericMatrix uMdraw(nkeeptrain,J);
  Rcpp::NumericMatrix uYdraw(nkeeptrain,J);
  
  //random number generation
  arn genM;
  arn genY;
  
  //heterbart mBM(numtree);
  //heterbart yBM(numtree);
  bart mBM(numtree);
  bart yBM(numtree);
  
  if(matXinfo.size()>0) {
    xinfo _matXxi;
    _matXxi.resize(pm);
    for(size_t q=0;q<pm;q++) {
      _matXxi[q].resize(matXnc[q]);
      for(int nc=0;nc<matXnc[q];nc++) _matXxi[q][nc]=matXinfo(q, nc);
    }
    mBM.setxinfo(_matXxi);
  }
  
  if(matMinfo.size()>0) {
    xinfo _matMxi;
    _matMxi.resize(py);
    for(size_t q=0;q<py;q++) {
      _matMxi[q].resize(matMnc[q]);
      for(int nc=0;nc<matMnc[q];nc++) _matMxi[q][nc]=matMinfo(q, nc);
    }
    yBM.setxinfo(_matMxi);
  }
#else
  
#define MDRAW(a, b) Mdraw[a][b]
#define YDRAW(a, b) Ydraw[a][b]  
#define UMDRAW(a, b) uMdraw[a][b]
#define UYDRAW(a, b) uYdraw[a][b]
  
  void crBARTmediation(int *typeM,     // 1:continuous, 2:binary, 3:multinomial
                       int *typeY,     // 1:continuous, 2:binary, 3:multinomial
                       size_t *n,      // number of observations in training data
                       size_t *pm,	    // dimension of matX
                       double *imatX, // matX, train, pm x n //(transposed so rows are contiguous in memory)
                       double *iM,    // M, train, n x 1
                       size_t *py,	    // dimension of matY
                       double *imatM, // matX, train, py x n //(transposed so rows are contiguous in memory)
                       double *iY,    // Y, train, n x 1
                       int *u_index,
                       int *n_j_vec,
                       double *uM,
                       double *uY,
                       size_t *J,
                       double *B_uM,
                       double *B_uY,
                       size_t *numtree, // number of trees
                       int *matXnc,    // number of cut points
                       int *matMnc,    // number of cut points
                       size_t *numdraw, // number of kept draws (except for thinnning ..)
                       size_t *burn,    // number of burn-in draws skipped
                       size_t *thin,    // thinning
                       double *mybeta,
                       double *alpha,
                       double *Moffset,
                       double *Yoffset,
                       double *Mtau,
                       double *Ytau,
                       double *nu,
                       double *Mlambda,
                       double *Ylambda,
                       double iMsigest,
                       double iYsigest,
                       bool *dart,        // dart prior: true(1)=yes, false(0)=no   
                       double *theta,
                       double *omega, 
                       double *a,	       // param a for sparsity prior                         
                       double *b,	       // param b for sparsity prior                        
                       double *matXrho,   // param matXrho for sparsity prior (default to pm)   
                       double *matMrho,   // param matMrho for sparsity prior (default to py)   
                       bool *aug,         // categorical strategy: true(1)=data augment  
                       size_t *printevery,
                       // additional parameters needed to call from C++
                       unsigned int *n1,
                       unsigned int *n2,
                       unsigned int *n3,
                       unsigned int *n4,
                       double *Msdraw,
                       double *Ysdraw,
                       double *_Mdraw,
                       double *_Ydraw,
                       double *_uMdraw,
                       double *_uYdraw) {
    //--------------------------------------------------
    //return data structures (using C++)
    size_t nkeeptrain=numdraw/thin, nkeeptreedraws=numdraw/thin;
    std::vector<double*> Mdraw(nkeeptrain);
    std::vector<double*> Ydraw(nkeeptrain);
    std::vector<double*> uMdraw(nkeeptrain);
    std::vector<double*> uYdraw(nkeeptrain);
    
    std::vector<double*> muMudraw(nkeeptrain);
    std::vector<double*> muYudraw(nkeeptrain);
    std::vector<double*> sigMudraw(nkeeptrain);
    std::vector<double*> sigYudraw(nkeeptrain);
    std::vector<double*> rhoMYudraw(nkeeptrain);
    
    for(size_t it=0; it<nkeeptrain; ++it) {
      Mdraw[it]=&_Mdraw[it*n];
      Ydraw[it]=&_Ydraw[it*n];
      uMdraw[it]=&_uMdraw[it*J];
      uYdraw[it]=&_uYdraw[it*J];
    }
    
    //matrix to return dart posteriors (counts and probs)
    std::vector< std::vector<size_t> > matXvarcnt;
    std::vector< std::vector<double> > matXvarprb;
    std::vector< std::vector<size_t> > matMvarcnt;
    std::vector< std::vector<double> > matMvarprb;
    
    //random number generation
    arn genM(n1, n2);
    arn genY(n3, n4);
    
    //heterbart mBM(numtree);
    //heterbart yBM(numtree);
    bart mBM(numtree);
    bart yBM(numtree);
#endif
    
    std::stringstream matXtreess;  //string stream to write trees to
    matXtreess.precision(15);
    matXtreess << nkeeptreedraws << "," << numtree << "," << pm << "," << endl;
    
    std::stringstream matMtreess;  //string stream to write trees to
    matMtreess.precision(15);
    matMtreess << nkeeptreedraws << "," << numtree << "," << py << "," << endl;
    
    printf("*****Calling rBARTmediation: typeM=%d\n", typeM);
    printf("*****Calling rBARTmediation: typeY=%d\n", typeY);
    
    size_t skiptr=thin, skiptreedraws=thin;
    
    //--------------------------------------------------
    // print args
    printf("*****Data:\n");
    printf("data:n,pm,py: %zu, %zu, %zu\n",n,pm,py);
    printf("m1,mn: %lf, %lf\n",iM[0],iM[n-1]);
    printf("y1,yn: %lf, %lf\n",iY[0],iY[n-1]);
    printf("matx1,matx[n*pm]: %lf, %lf\n",imatX[0],imatX[n*pm-1]);
    printf("matm1,matm[n*py]: %lf, %lf\n",imatM[0],imatM[n*py-1]);
    //   if(hotdeck) 
    //printf("warning: missing elements in x multiply imputed with hot decking\n");
    printf("*****Number of Trees: %zu\n",numtree);
    printf("*****Number of Cut Points (matX): %d ... %d\n", matXnc[0], matXnc[pm-1]);
    printf("*****Number of Cut Points (matM): %d ... %d\n", matMnc[0], matMnc[py-1]);
    printf("*****burn,numdraw,thin: %zu,%zu,%zu\n",burn,numdraw,thin);
    cout << "*****Prior:beta,alpha,Mtau,nu,Mlambda,Moffset: " 
         << mybeta << ',' << alpha << ',' << Mtau << ',' 
         << nu << ',' << Mlambda << ',' << Moffset << endl;
    cout << "*****Prior:beta,alpha,Ytau,nu,Ylambda,Yoffset: " 
         << mybeta << ',' << alpha << ',' << Ytau << ',' 
         << nu << ',' << Ylambda << ',' << Yoffset << endl;
    printf("*****iMsigest: %lf\n",iMsigest);
    printf("*****iYsigest: %lf\n",iYsigest);
    cout << "*****Dirichlet:sparse,theta,omega,a,b,matXrho,augment: " 
         << dart << ',' << theta << ',' << omega << ',' << a << ',' 
         << b << ',' << matXrho << ',' << aug << endl;
    cout << "*****Dirichlet:sparse,theta,omega,a,b,matMrho,augment: " 
         << dart << ',' << theta << ',' << omega << ',' << a << ',' 
         << b << ',' << matMrho << ',' << aug << endl;
    printf("*****printevery: %zu\n",printevery);
    
    //--------------------------------------------------
    //create temporaries
    double df = n + nu;
    double *Mz = new double[n];
    double *Yz = new double[n];
    double *Msign; if(typeM!=1) Msign = new double[n];
    double *Ysign; if(typeY!=1) Ysign = new double[n];
    for(size_t i=0; i<n; i++) {
      if(typeM==1) {
        Mz[i] = iM[i] - Moffset;
      } else {
        if(iM[i]==0) {
          Msign[i] = -1.;
        } else {
          Msign[i] = 1.;
        }
        Mz[i] = Msign[i];
      }
      if(typeY==1) {
        Yz[i] = iY[i] - Yoffset;
      } else {
        if(iY[i]==0) {
          Ysign[i] = -1.;
        } else {
          Ysign[i] = 1.;
        }
        Yz[i] = Ysign[i];
      }
    }
    
    //--------------------------------------------------
    // double *uM = new double[J];
    // double *uY = new double[J];
    double mu_uM = 0., sd_uM = B_uM; // , sd_uM = B_uM * 0.5, tau_uM=pow(sd_uM, -2.), invB2M=pow(B_uM, -2.);
    double mu_uY = 0., sd_uY = B_uY; // , sd_uY = B_uY * 0.5, tau_uY=pow(sd_uY, -2.), invB2Y=pow(B_uY, -2.);
    if(uM[0]!=uM[0] || uY[0]!=uY[0]) {
      for(size_t j=0; j<J; j++) {
        uM[j]=sd_uM * genM.normal() + mu_uM;
        uY[j]=sd_uY * genY.normal() + mu_uY;
      }
    }
    
    arma::mat eye_mat_22 = arma::eye(2,2);
    arma::mat zero_vec_2 = arma::zeros(2);
    arma::mat zero_mat_22 = arma::zeros(2,2);
    
    double nu_uMY0 = 4; // > p - 1 (p=2)
    double lambda_uMY0 = 0.5;
    double nu_uMY = nu_uMY0 + J;
    double lambda_uMY = lambda_uMY0 + J;
    
    arma::vec MU_uMY0 = zero_vec_2;
    // arma::mat SIG_uMY0 = eye_mat_22;
    // arma::mat SIG_uMY0 = {{B_uM,0},{0,B_uY}};
    // arma::mat SIG_uMY0 = {{B_uM/2,0},{0,B_uY/2}};
    arma::mat SIG_uMY0 = {{B_uM/J,0},{0,B_uY/J}};
    // arma::mat SIG_uMY0 = {{2*B_uM/J,0},{0,2*B_uY/J}};
    // arma::mat SIG_uMY0 = {{1/B_uM,0},{0,1/B_uY}};
    arma::mat invSIG_uMY0 =  inv(SIG_uMY0);
    arma::vec MU_uMYtmp;
    arma::mat SIG_uMYtmp;
    arma::mat invSIG_uMYtmp; // inv(SIG_uMYtmp);
    arma::vec MU_uMYstr;
    arma::mat SIG_uMYstr;
    arma::mat invSIG_uMYstr; // inv(SIG_uMYstr);
    arma::vec MU_uMYprop;
    arma::mat SIG_uMYprop;
    arma::mat invSIG_uMYprop; // inv(SIG_uMYprop);
    
    // for(size_t j=0; j<J; j++) {
    //   arma::vec tmp = arma::mvnrnd(MU_uMY0, SIG_uMY0/lambda_uMY0); // 
    //   uM[j] = tmp(0);
    //   uY[j] = tmp(1);
    // }
    
    //--------------------------------------------------
    // draw MU_uMYstr & SIG_uMYstr
    arma::vec uMYbar = zero_vec_2;
    for(size_t j=0; j<J; j++) {
      uMYbar(0) += uM[j];
      uMYbar(1) += uY[j];
    }
    uMYbar /= J;
    arma::mat uMYtuMY = zero_mat_22;
    for(size_t j=0; j<J; j++) {
      uMYtuMY(0,0) += pow(uM[j] - uMYbar(0), 2.);
      uMYtuMY(1,1) += pow(uY[j] - uMYbar(1), 2.);
      uMYtuMY(0,1) += ((uM[j] - uMYbar(0)) * (uY[j] - uMYbar(1)));
    }
    uMYtuMY(1,0) = uMYtuMY(0,1);
    SIG_uMYtmp = SIG_uMY0 + uMYtuMY + (lambda_uMY0*J/lambda_uMY)*(uMYbar*uMYbar.t()); // MU_uMY0 = 0
    SIG_uMYstr = iwishrnd(SIG_uMYtmp, nu_uMY);
    invSIG_uMYstr = inv(SIG_uMYstr);
    MU_uMYtmp = J * uMYbar/lambda_uMY; // (lambda_uMY0 * MU_uMY0 + J * uMYbar)/(lambda_uMY0 + J);
    MU_uMYstr = mvnrnd(MU_uMYtmp, SIG_uMYstr/lambda_uMY);
    
    // arma::mat uMYtuMY = zero_mat_22;
    // for(size_t j=0; j<J; j++) {
    //   uMYtuMY(0,0) += pow(uM[j], 2.);
    //   uMYtuMY(1,1) += pow(uY[j], 2.);
    //   uMYtuMY(0,1) += (uM[j] * uY[j]);
    // }
    // uMYtuMY(1,0) = uMYtuMY(0,1);
    // SIG_uMYtmp = SIG_uMY0 + uMYtuMY;
    // SIG_uMYstr = iwishrnd(SIG_uMYtmp, nu_uMY);
    // invSIG_uMYstr = inv(SIG_uMYstr);
    // MU_uMYstr = zero_vec_2;
    
    //--------------------------------------------------
    // double ratio;
    // double Mzlik = R_NegInf;
    // double Yzlik = R_NegInf;
    // double *uMYlik_j = new double[J];
    // for(size_t j=0; j<J; j++) {
    //   uMYlik_j[j] = R_NegInf;
    // }
    
    // set up BART model
    mBM.setprior(alpha,mybeta,Mtau);
    // mBM.setprior(1-(1-alpha)/2,mybeta/2,Mtau);
    mBM.setdata(pm,n,imatX,Mz,matXnc);
    mBM.setdart(a,b,matXrho,aug,dart);
    
    yBM.setprior(alpha,mybeta,Ytau);
    // yBM.setprior(1-(1-alpha)/2,mybeta/2,Ytau);
    yBM.setdata(py,n,imatM,Yz,matMnc);
    yBM.setdart(a,b,matMrho,aug,dart);
    
    // dart iterations
    std::vector<double> imatXvarprb (pm,0.);
    std::vector<size_t> imatXvarcnt (pm,0);
    
    std::vector<double> imatMvarprb (py,0.);
    std::vector<size_t> imatMvarcnt (py,0);
    
    //--------------------------------------------------
    // mcmc
    printf("\nMCMC\n");
    // size_t index;
    size_t trcnt=0; // count kept train draws
    bool keeptreedraw, typeM1=(typeM==1 && Mlambda!=0.), typeY1=(typeY==1 && Ylambda!=0.);
    
    time_t tp;
    int time1 = time(&tp);
    size_t total=numdraw+burn;
    xinfo& matXxi = mBM.getxinfo();
    xinfo& matMxi = yBM.getxinfo();
    
    for(size_t postrep=0;postrep<total;postrep++) {
      //--------------------------------------------------
      if(postrep==(burn/2)&&dart) {
        mBM.startdart();
        yBM.startdart();
      }
      
      // //--------------------------------------------------
      // double *Mzprop = new double[n];
      // double Mzlik_prop = 0.;
      // for(size_t i=0;i<n;i++) {
      //   if(typeM==1){
      //     Mzprop[i] = iM[i] - (Moffset+uM[u_index[i]]); // (MOffset+uM[u_index[i]])
      //     Mzlik_prop += R::dnorm(Mzprop[i], 0., iMsigest, true);
      //   } else if(typeM==2){
      //     Mzprop[i] = Msign[i] * rtnorm(Msign[i]*mBM.f(i), -Msign[i]*(Moffset+uM[u_index[i]]), 1., genM);
      //     Mzlik_prop += R::pnorm(Msign[i] * (Moffset+mBM.f(i)+uM[u_index[i]]), 0., 1., true, true);
      //   }
      // }
      // // acceptance ratio
      // ratio = exp(Mzlik_prop-Mzlik);
      // if (ratio > genM.uniform()){
      //   Mzlik = Mzlik_prop;
      //   Mz = Mzprop;
      // }
      // 
      // //--------------------------------------------------
      // double *Yzprop = new double[n];
      // double Yzlik_prop = 0.;
      // for(size_t i=0;i<n;i++) {
      //   if(typeY==1){
      //     Yzprop[i] = iY[i] - (Yoffset+uY[u_index[i]]); // YOffset+uY[u_index[i]])
      //     Yzlik_prop += R::dnorm(Yzprop[i], 0., iYsigest, true);
      //   } else if(typeY==2){
      //     Yzprop[i] = Ysign[i] * rtnorm(Ysign[i]*yBM.f(i), -Ysign[i]*(Yoffset+uY[u_index[i]]), 1., genY);
      //     Yzlik_prop += R::pnorm(Ysign[i] * (Yoffset+yBM.f(i)+uY[u_index[i]]), 0., 1., true, true);
      //   }
      // }
      // // acceptance ratio
      // ratio = exp(Yzlik_prop-Yzlik);
      // if (ratio > genM.uniform()){
      //   Yzlik = Yzlik_prop;
      //   Yz = Yzprop;
      // }
      
      //--------------------------------------------------
      for(size_t i=0;i<n;i++) {
        if(typeM==1){
          Mz[i] = iM[i] - (Moffset+uM[u_index[i]]); // +uM[u_index[i]] // +MU_uMYstr(0)
        } else if(typeM==2){
          Mz[i] = Msign[i] * rtnorm(Msign[i]*mBM.f(i), -Msign[i]*(Moffset+uM[u_index[i]]), 1., genM);
        }
        if(typeY==1){
          Yz[i] = iY[i] - (Yoffset+uY[u_index[i]]); // +uY[u_index[i]] // +MU_uMYstr(1)
        } else if(typeY==2){
          Yz[i] = Ysign[i] * rtnorm(Ysign[i]*yBM.f(i), -Ysign[i]*(Yoffset+uY[u_index[i]]), 1., genY);
        }
      }
      
      //--------------------------------------------------
      mBM.draw(iMsigest, genM);
      yBM.draw(iYsigest, genY);
      
      //--------------------------------------------------
      if(typeM1){
        double Mrss = 0.;
        for(size_t i=0;i<n;i++) {
          Mrss += pow((iM[i]-(Moffset+mBM.f(i)+uM[u_index[i]])), 2.); // +uM[u_index[i]]
        }
        iMsigest = sqrt((nu*Mlambda + Mrss)/genM.chi_square(df));
      }
      if(typeY1){
        double Yrss = 0.;
        for(size_t i=0;i<n;i++) {
          Yrss += pow((iY[i]-(Yoffset+yBM.f(i)+uY[u_index[i]])), 2.); // +uY[u_index[i]]
        }
        iYsigest = sqrt((nu*Ylambda + Yrss)/genY.chi_square(df));
      }
      
      //--------------------------------------------------
      // draw MU_uMYstr & SIG_uMYstr
      uMYbar = zero_vec_2;
      for(size_t j=0; j<J; j++) {
        uMYbar(0) += uM[j];
        uMYbar(1) += uY[j];
      }
      uMYbar /= J;
      uMYtuMY = zero_mat_22;
      for(size_t j=0; j<J; j++) {
        uMYtuMY(0,0) += pow(uM[j] - uMYbar(0), 2.);
        uMYtuMY(1,1) += pow(uY[j] - uMYbar(1), 2.);
        uMYtuMY(0,1) += ((uM[j] - uMYbar(0)) * (uY[j] - uMYbar(1)));
      }
      uMYtuMY(1,0) = uMYtuMY(0,1);
      SIG_uMYtmp = SIG_uMY0 + uMYtuMY + (lambda_uMY0*J/lambda_uMY)*(uMYbar*uMYbar.t()); // MU_uMY0 = 0
      SIG_uMYstr = iwishrnd(SIG_uMYtmp, nu_uMY);
      invSIG_uMYstr = inv(SIG_uMYstr);
      MU_uMYtmp = J * uMYbar/lambda_uMY; // (lambda_uMY0 * MU_uMY0 + J * uMYbar)/(lambda_uMY0 + J);
      MU_uMYstr = mvnrnd(MU_uMYtmp, SIG_uMYstr/lambda_uMY);
      
      // uMYtuMY = zero_mat_22;
      // for(size_t j=0; j<J; j++) {
      //   uMYtuMY(0,0) += pow(uM[j], 2.);
      //   uMYtuMY(1,1) += pow(uY[j], 2.);
      //   uMYtuMY(0,1) += (uM[j] * uY[j]);
      // }
      // uMYtuMY(1,0) = uMYtuMY(0,1);
      // SIG_uMYtmp = SIG_uMY0 + uMYtuMY;
      // SIG_uMYstr = iwishrnd(SIG_uMYtmp, nu_uMY);
      // invSIG_uMYstr = inv(SIG_uMYstr);
      // MU_uMYstr = zero_vec_2;
      
      //--------------------------------------------------
      // draw uM, uY
      size_t n_j, ii; // , ii_j;
      double precM = pow(iMsigest, -2.), precY = pow(iYsigest, -2.);
      
      // //--------------------------------------------------
      // double *uMYlik_j_prop = new double[J];
      // for(size_t j=0; j<J; j++) {
      //   uMYlik_j_prop[j] = R_NegInf;
      // }
      
      ii=0;
      if (typeM==1 && typeY==1) {
        //--------------------------------------------------
        //--------------------------------------------------
        for(size_t j=0; j<J; j++) {
          n_j = n_j_vec[j];
          
          invSIG_uMYprop = invSIG_uMYstr;
          invSIG_uMYprop(0,0) += n_j*precM;
          invSIG_uMYprop(1,1) += n_j*precY;
          SIG_uMYprop = inv(invSIG_uMYprop);
          
          MU_uMYprop = zero_vec_2;
          for(size_t itmp=0; itmp<n_j; itmp++) {
            MU_uMYprop(0) += (iM[ii]-(Moffset+mBM.f(ii)));
            MU_uMYprop(1) += (iY[ii]-(Yoffset+yBM.f(ii)));
            ii++;
          }
          MU_uMYprop(0) *= precM;
          MU_uMYprop(1) *= precY;
          MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          
          arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          uM[j] = uMYprop(0);
          uY[j] = uMYprop(1);
          
          // n_j = n_j_vec[j];
          // 
          // invSIG_uMYprop = invSIG_uMYstr;
          // invSIG_uMYprop(0,0) += n_j*precM;
          // invSIG_uMYprop(1,1) += n_j*precY;
          // SIG_uMYprop = inv(invSIG_uMYprop);
          // 
          // ii_j = ii;
          // MU_uMYprop = zero_vec_2;
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   MU_uMYprop(0) += (iM[ii_j]-(Moffset+mBM.f(ii_j)));
          //   MU_uMYprop(1) += (iY[ii_j]-(Yoffset+yBM.f(ii_j)));
          //   ii_j++;
          // }
          // MU_uMYprop(0) *= precM;
          // MU_uMYprop(1) *= precY;
          // MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          // 
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          // 
          // ii_j = ii;
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] = as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::dnorm(iM[ii_j], (Moffset + mBM.f(ii_j) + uMYprop(0)), iMsigest, true) +
          //     R::dnorm(iY[ii_j], (Yoffset + yBM.f(ii_j) + uMYprop(1)), iYsigest, true);
          //   ii_j++; ii++;
          // }
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
        }
      } else if (typeM==2 && typeY==1) {
        //--------------------------------------------------
        //--------------------------------------------------
        for(size_t j=0; j<J; j++) {
          n_j = n_j_vec[j];
          
          invSIG_uMYprop = invSIG_uMYstr;
          invSIG_uMYprop(0,0) += n_j*precM;
          invSIG_uMYprop(1,1) += n_j*precY;
          SIG_uMYprop = inv(invSIG_uMYprop);
          
          MU_uMYprop = zero_vec_2;
          for(size_t itmp=0; itmp<n_j; itmp++) {
            MU_uMYprop(0) += (Mz[ii]-(mBM.f(ii))); // use latent variable for binary M
            MU_uMYprop(1) += (iY[ii]-(Yoffset+yBM.f(ii)));
            ii++;
          }
          MU_uMYprop(0) *= precM;
          MU_uMYprop(1) *= precY;
          MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          
          arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          uM[j] = uMYprop(0);
          uY[j] = uMYprop(1);
          
          // n_j = n_j_vec[j];
          // 
          // invSIG_uMYprop = invSIG_uMYstr;
          // invSIG_uMYprop(0,0) += n_j*precM;
          // invSIG_uMYprop(1,1) += n_j*precY;
          // SIG_uMYprop = inv(invSIG_uMYprop);
          // 
          // ii_j = ii;
          // MU_uMYprop = zero_vec_2;
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   MU_uMYprop(0) += (Mz[ii_j]-(mBM.f(ii_j))); // use latent variable for binary M
          //   MU_uMYprop(1) += (iY[ii_j]-(Yoffset+yBM.f(ii_j)));
          //   ii_j++;
          // }
          // MU_uMYprop(0) *= precM;
          // MU_uMYprop(1) *= precY;
          // MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          // 
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          // 
          // ii_j = ii;
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] = as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::pnorm(Msign[ii] * (Moffset + mBM.f(ii) + uMYprop(0)), 0., 1., true, true) +
          //     R::dnorm(iY[ii], (Yoffset + yBM.f(ii) + uMYprop(1)), iYsigest, true);
          //   ii_j++; ii++;
          // }
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
          
          // // Proposal
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYstr, SIG_uMYstr);
          // 
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] += as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::pnorm(Msign[ii] * (Moffset + mBM.f(ii) + uMYprop(0)), 0., 1., true, true) +
          //     R::dnorm(iY[ii], (Yoffset + yBM.f(ii) + uMYprop(1)), iYsigest, true);
          //   ii++;
          // }
          // 
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
        }
      } else if (typeM==1 && typeY==2) {
        //--------------------------------------------------
        //--------------------------------------------------
        for(size_t j=0; j<J; j++) {
          n_j = n_j_vec[j];
          
          invSIG_uMYprop = invSIG_uMYstr;
          invSIG_uMYprop(0,0) += n_j*precM;
          invSIG_uMYprop(1,1) += n_j*precY;
          SIG_uMYprop = inv(invSIG_uMYprop);
          
          MU_uMYprop = zero_vec_2;
          for(size_t itmp=0; itmp<n_j; itmp++) {
            MU_uMYprop(0) += (iM[ii]-(Moffset+mBM.f(ii)));
            MU_uMYprop(1) += (Yz[ii]-(yBM.f(ii))); // use latent variable for binary M
            ii++;
          }
          MU_uMYprop(0) *= precM;
          MU_uMYprop(1) *= precY;
          MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          
          arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          uM[j] = uMYprop(0);
          uY[j] = uMYprop(1);
          
          // n_j = n_j_vec[j];
          // 
          // invSIG_uMYprop = invSIG_uMYstr;
          // invSIG_uMYprop(0,0) += n_j*precM;
          // invSIG_uMYprop(1,1) += n_j*precY;
          // SIG_uMYprop = inv(invSIG_uMYprop);
          // 
          // ii_j = ii;
          // MU_uMYprop = zero_vec_2;
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   MU_uMYprop(0) += (iM[ii_j]-(Moffset+mBM.f(ii_j)));
          //   MU_uMYprop(1) += (Yz[ii_j]-(yBM.f(ii_j))); // use latent variable for binary M
          //   ii_j++;
          // }
          // MU_uMYprop(0) *= precM;
          // MU_uMYprop(1) *= precY;
          // MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          // 
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          // 
          // ii_j = ii;
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] = as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::dnorm(iM[ii], (Moffset + mBM.f(ii) + uMYprop(0)), iMsigest, true) +
          //     R::pnorm(Ysign[ii] * (Yoffset + yBM.f(ii) + uMYprop(1)), 0., 1., true, true);
          //   ii_j++; ii++;
          // }
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
          
          // // Proposal
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYstr, SIG_uMYstr);
          // 
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] += as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::dnorm(iM[ii], (Moffset + mBM.f(ii) + uMYprop(0)), iMsigest, true) +
          //     R::pnorm(Ysign[ii] * (Yoffset + yBM.f(ii) + uMYprop(1)), 0., 1., true, true);
          //   ii++;
          // }
          // 
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
        }
      } else if (typeM==2 && typeY==2) {
        //--------------------------------------------------
        //--------------------------------------------------
        for(size_t j=0; j<J; j++) {
          n_j = n_j_vec[j];
          
          invSIG_uMYprop = invSIG_uMYstr;
          invSIG_uMYprop(0,0) += n_j*precM;
          invSIG_uMYprop(1,1) += n_j*precY;
          SIG_uMYprop = inv(invSIG_uMYprop);
          
          MU_uMYprop = zero_vec_2;
          for(size_t itmp=0; itmp<n_j; itmp++) {
            MU_uMYprop(0) += (Mz[ii]-(mBM.f(ii))); // use latent variable for binary M
            MU_uMYprop(1) += (Yz[ii]-(yBM.f(ii))); // use latent variable for binary M
            ii++;
          }
          MU_uMYprop(0) *= precM;
          MU_uMYprop(1) *= precY;
          MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          
          arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          uM[j] = uMYprop(0);
          uY[j] = uMYprop(1);
          
          // n_j = n_j_vec[j];
          // 
          // invSIG_uMYprop = invSIG_uMYstr;
          // invSIG_uMYprop(0,0) += n_j*precM;
          // invSIG_uMYprop(1,1) += n_j*precY;
          // SIG_uMYprop = inv(invSIG_uMYprop);
          // 
          // ii_j = ii;
          // MU_uMYprop = zero_vec_2;
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   MU_uMYprop(0) += (Mz[ii_j]-(mBM.f(ii_j))); // use latent variable for binary M
          //   MU_uMYprop(1) += (Yz[ii_j]-(yBM.f(ii_j))); // use latent variable for binary M
          //   ii_j++;
          // }
          // MU_uMYprop(0) *= precM;
          // MU_uMYprop(1) *= precY;
          // MU_uMYprop = SIG_uMYprop * (invSIG_uMYstr * MU_uMYstr + MU_uMYprop);
          // 
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYprop, SIG_uMYprop);
          // 
          // ii_j = ii;
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] = as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::pnorm(Msign[ii] * (Moffset + mBM.f(ii) + uMYprop(0)), 0., 1., true, true) +
          //     R::pnorm(Ysign[ii] * (Yoffset + yBM.f(ii) + uMYprop(1)), 0., 1., true, true);
          //   ii_j++; ii++;
          // }
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
          
          // // Proposal
          // arma::vec uMYprop = arma::mvnrnd(MU_uMYstr, SIG_uMYstr);
          // 
          // // uMYlik_j_prop
          // uMYlik_j_prop[j] += as_scalar(dmvnorm(uMYprop.t(), MU_uMYstr, SIG_uMYstr, true));
          // for(size_t itmp=0; itmp<n_j; itmp++) {
          //   uMYlik_j_prop[j] +=
          //     R::pnorm(Msign[ii] * (Moffset + mBM.f(ii) + uMYprop(0)), 0., 1., true, true) +
          //     R::pnorm(Ysign[ii] * (Yoffset + yBM.f(ii) + uMYprop(1)), 0., 1., true, true);
          //   ii++;
          // }
          // 
          // // acceptance ratio
          // ratio = exp(uMYlik_j_prop[j]-uMYlik_j[j]);
          // if (ratio > genM.uniform()){
          //   uMYlik_j[j] = uMYlik_j_prop[j];
          //   uM[j] = uMYprop(0);
          //   uY[j] = uMYprop(1);
          // }
        }
      }
      
      //--------------------------------------------------
      if(postrep>=burn) {
        if(postrep%printevery==0) {
          printf("done %zu (out of %zu)\n",postrep,numdraw+burn);
        }
        if(nkeeptrain && (((postrep-burn+1) % skiptr) == 0)) {
          for(size_t i=0;i<n;i++) {
            MDRAW(trcnt,i) = Moffset + mBM.f(i);
            YDRAW(trcnt,i) = Yoffset + yBM.f(i);
          }
          Msdraw[trcnt]=iMsigest;
          Ysdraw[trcnt]=iYsigest;
          for(size_t j=0;j<J;j++) {
            UMDRAW(trcnt,j) = uM[j];
            UYDRAW(trcnt,j) = uY[j];
          }
          muMudraw[trcnt] = MU_uMYstr(0);
          muYudraw[trcnt] = MU_uMYstr(1);
          sigMudraw[trcnt] = sqrt(SIG_uMYstr(0,0));
          sigYudraw[trcnt] = sqrt(SIG_uMYstr(1,1));
          rhoMYudraw[trcnt] = SIG_uMYstr(0,1)/sqrt(SIG_uMYstr(0,0)*SIG_uMYstr(1,1));
          
          trcnt+=1;
          
          keeptreedraw = nkeeptreedraws && (((postrep-burn+1) % skiptreedraws) == 0);
          if(keeptreedraw) {
            matXtreess << ",";
            matMtreess << ",";
            for(size_t num=0;num<numtree;num++) {
              matXtreess << mBM.gettree(num);
              matMtreess << yBM.gettree(num);
#ifndef NoRcpp
              size_t it=(postrep-burn)/skiptreedraws;
              
              imatXvarcnt=mBM.getnv();
              imatXvarprb=mBM.getpv();
              for(size_t q=0;q<pm;q++){
                matXvarcnt(it,q)=imatXvarcnt[q];
                matXvarprb(it,q)=imatXvarprb[q];
              }
              
              imatMvarcnt=yBM.getnv();
              imatMvarprb=yBM.getpv();
              for(size_t q=0;q<py;q++){
                matMvarcnt(it,q)=imatMvarcnt[q];
                matMvarprb(it,q)=imatMvarprb[q];
              }
#else
              matXvarcnt.push_back(mBM.getnv());
              matXvarprb.push_back(mBM.getpv());
              matMvarcnt.push_back(yBM.getnv());
              matMvarprb.push_back(yBM.getpv());
#endif
            }
          }
        }
      }
    }
    int time2 = time(&tp);
    printf("time: %ds\n",time2-time1);
    printf("trcnt: %zu\n",trcnt);
    
    delete[] Mz;
    delete[] Yz;
    if(typeM!=1) delete[] Msign;
    if(typeY!=1) delete[] Ysign;
    
#ifndef NoRcpp
    
    //--------------------------------------------------
    //return list
    Rcpp::List ret;
    
    ret["iMsigest"]=Msdraw;
    ret["iYsigest"]=Ysdraw;
    ret["Mdraw"]=Mdraw;
    ret["Ydraw"]=Ydraw;
    ret["uMdraw"]=uMdraw;
    ret["uYdraw"]=uYdraw;
    ret["mu.uM"]=muMudraw;
    ret["mu.uY"]=muYudraw;
    ret["sig.uM"]=sigMudraw;
    ret["sig.uY"]=sigYudraw;
    ret["rho.uMY"]=rhoMYudraw;
    ret["matXvarcount"]=matXvarcnt;
    ret["matMvarcount"]=matMvarcnt;
    ret["matXvarprob"]=matXvarprb;
    ret["matMvarprob"]=matMvarprb;
    
    Rcpp::List matXiret(matXxi.size());
    for(size_t it=0;it<matXxi.size();it++) {
      Rcpp::NumericVector matXtmp(matXxi[it].size());
      std::copy(matXxi[it].begin(), matXxi[it].end(), matXtmp.begin());
      matXiret[it] = Rcpp::NumericVector(matXtmp);
    }
    Rcpp::List matMiret(matMxi.size());
    for(size_t it=0;it<matMxi.size();it++) {
      Rcpp::NumericVector matMtmp(matMxi[it].size());
      std::copy(matMxi[it].begin(), matMxi[it].end(), matMtmp.begin());
      matMiret[it] = Rcpp::NumericVector(matMtmp);
    }
    
    Rcpp::List matXtreesL;
    matXtreesL["cutpoints"] = matXiret;
    matXtreesL["trees"]=Rcpp::CharacterVector(matXtreess.str());
    ret["matXtreedraws"] = matXtreesL;
    Rcpp::List matMtreesL;
    matMtreesL["cutpoints"] = matMiret;
    matMtreesL["trees"]=Rcpp::CharacterVector(matMtreess.str());
    ret["matMtreedraws"] = matMtreesL;
    
    return ret;
#else
    
#endif
    
  }