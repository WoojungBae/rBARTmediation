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
                                SEXP _MOffset,
                                SEXP _YOffset,
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
  double MOffset = Rcpp::as<double>(_MOffset);
  double YOffset = Rcpp::as<double>(_YOffset);
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
  Rcpp::NumericVector rhoYMudraw(nkeeptrain);
  Rcpp::NumericVector sdMudraw(nkeeptrain);
  Rcpp::NumericVector sdYudraw(nkeeptrain);
  Rcpp::NumericVector muMudraw(nkeeptrain);
  Rcpp::NumericVector muYudraw(nkeeptrain);
  Rcpp::NumericMatrix Mdraw(nkeeptrain,n);
  Rcpp::NumericMatrix Ydraw(nkeeptrain,n);
  Rcpp::NumericMatrix uMdraw(nkeeptrain,J);
  Rcpp::NumericMatrix uYdraw(nkeeptrain,J);
  
  //random number generation
  arn gen;
  
  //heterbart mBM(numtree);
  //heterbart yBM(numtree);
  bart mBM(numtree);
  bart yBM(numtree);
  
  if(matXinfo.size()>0) {
    xinfo _matXxi;
    _matXxi.resize(pm);
    for(size_t q=0;q<pm;q++) {
      _matXxi[q].resize(matXnc[q]);
      for(size_t j=0;j<matXnc[q];j++) _matXxi[q][j]=matXinfo(q, j);
    }
    mBM.setxinfo(_matXxi);
  }
  
  if(matMinfo.size()>0) {
    xinfo _matMxi;
    _matMxi.resize(py);
    for(size_t q=0;q<py;q++) {
      _matMxi[q].resize(matMnc[q]);
      for(size_t j=0;j<matMnc[q];j++) _matMxi[q][j]=matMinfo(q, j);
    }
    yBM.setxinfo(_matMxi);
  }
#else
  
#define MDRAW(a, b) Mdraw[a][b]
#define YDRAW(a, b) Ydraw[a][b]  
#define UMDRAW(a, b) uMdraw[a][b]
#define UYDRAW(a, b) uYdraw[a][b]
  
  void crBARTmediation(int typeM,     // 1:continuous, 2:binary, 3:multinomial
                       int typeY,     // 1:continuous, 2:binary, 3:multinomial
                       size_t n,      // number of observations in training data
                       size_t pm,	    // dimension of matX
                       double* imatX, // matX, train, pm x n //(transposed so rows are contiguous in memory)
                       double* iM,    // M, train, n x 1
                       size_t py,	    // dimension of matY
                       double* imatM, // matX, train, py x n //(transposed so rows are contiguous in memory)
                       double* iY,    // Y, train, n x 1
                       int *u_index,
                       int *n_j_vec,
                       double *uM,
                       double *uY,
                       size_t J,
                       double B_uM,
                       double B_uY,
                       size_t numtree, // number of trees
                       int *matXnc,    // number of cut points
                       int *matMnc,    // number of cut points
                       size_t numdraw, // number of kept draws (except for thinnning ..)
                       size_t burn,    // number of burn-in draws skipped
                       size_t thin,    // thinning
                       double mybeta,
                       double alpha,
                       double MOffset,
                       double YOffset,
                       double Mtau,
                       double Ytau,
                       double nu,
                       double Mlambda,
                       double Ylambda,
                       double iMsigest,
                       double iYsigest,
                       bool dart,        // dart prior: true(1)=yes, false(0)=no   
                       double theta,
                       double omega, 
                       double a,	       // param a for sparsity prior                         
                       double b,	       // param b for sparsity prior                        
                       double matXrho,   // param matXrho for sparsity prior (default to pm)   
                       double matMrho,   // param matMrho for sparsity prior (default to py)   
                       bool aug,         // categorical strategy: true(1)=data augment  
                       size_t printevery,
                       // additional parameters needed to call from C++
                       unsigned int n1,
                       unsigned int n2,
                       double* Msdraw,
                       double* Ysdraw,
                       double* sdMudraw,
                       double* sdYudraw,
                       double* _mdraw,
                       double* _ydraw,
                       double* _udraw) {
    //--------------------------------------------------
    //return data structures (using C++)
    size_t nkeeptrain=numdraw/thin, nkeeptreedraws=numdraw/thin;
    std::vector<double*> Mdraw(nkeeptrain);
    std::vector<double*> Ydraw(nkeeptrain);
    std::vector<double*> uMdraw(nkeeptrain);
    std::vector<double*> uYdraw(nkeeptrain);
    
    for(size_t it=0; it<nkeeptrain; ++it) {
      Mdraw[it]=&_mdraw[it*n];
      Ydraw[it]=&_ydraw[it*n];
      uMdraw[it]=&_udraw[it*J];
      uYdraw[it]=&_udraw[it*J];
    }
    
    //matrix to return dart posteriors (counts and probs)
    std::vector< std::vector<size_t> > matXvarcnt;
    std::vector< std::vector<double> > matXvarprb;
    std::vector< std::vector<size_t> > matMvarcnt;
    std::vector< std::vector<double> > matMvarprb;
    
    //random number generation
    arn gen(n1, n2);
    
    //heterbart mBM(numtree);
    //heterbart yBM(numtree);
    bart mBM(numtree);
    bart yBM(numtree);
#endif
    
    std::stringstream matXtreess;  //string stream to write trees to
    matXtreess.precision(10);
    matXtreess << nkeeptreedraws << "," << numtree << "," << pm << "," << endl;
    
    std::stringstream matMtreess;  //string stream to write trees to
    matMtreess.precision(10);
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
         << nu << ',' << Mlambda << ',' << MOffset << endl;
    cout << "*****Prior:beta,alpha,Ytau,nu,Ylambda,Yoffset: " 
         << mybeta << ',' << alpha << ',' << Mtau << ',' 
         << nu << ',' << Ylambda << ',' << YOffset << endl;
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
        Mz[i] = iM[i] - MOffset;
      } else {
        if(iM[i]==0) {
          Msign[i] = -1.;
        } else {
          Msign[i] = 1.;
        }
        Mz[i] = Msign[i];
      }
      if(typeY==1) {
        Yz[i] = iY[i] - YOffset;
      } else {
        if(iY[i]==0) {
          Ysign[i] = -1.;
        } else {
          Ysign[i] = 1.;
        }
        Yz[i] = Ysign[i];
      }
    }
    
    // double *uM = new double[J];
    // double *uY = new double[J];
    double mu_uM = 0., sd_uM = B_uM * 0.5, tau_uM=pow(sd_uM, -2.), invB2M=pow(B_uM, -2.);
    double mu_uY = 0., sd_uY = B_uY * 0.5, tau_uY=pow(sd_uY, -2.), invB2Y=pow(B_uY, -2.);
    if(uM[0]!=uM[0] || uY[0]!=uY[0]) {
      for(size_t j=0; j<J; j++) {
        uM[j]=sd_uM * gen.normal();
        uY[j]=sd_uY * gen.normal();
      }
    }
    
    double ratio;
    double rho_uYM = gen.uniform() * 2 - 1; // gen.uniform();
    double YMzlik = R_NegInf;
    double *uYMlik_j = new double[J];
    for(size_t j=0; j<J; j++) {
      uYMlik_j[j] = R_NegInf;
    }
    
    // set up BART model
    mBM.setprior(alpha,mybeta,Mtau);
    mBM.setdata(pm,n,imatX,Mz,matXnc);
    mBM.setdart(a,b,matXrho,aug,dart);
    
    yBM.setprior(alpha,mybeta,Ytau);
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
    int time1 = time(&tp), total=numdraw+burn;
    xinfo& matXxi = mBM.getxinfo();
    xinfo& matMxi = yBM.getxinfo();
    
    for(size_t postrep=0;postrep<total;postrep++) {
      if(postrep==(burn/2)&&dart) {
        mBM.startdart();
        yBM.startdart();
      }
      mBM.draw(iMsigest,gen);
      yBM.draw(iYsigest,gen);
      
      //--------------------------------------------------
      //--------------------------------------------------
      double *Mzprop = new double[n];
      double *Yzprop = new double[n];
      double YMzlik_prop = 0.;
      for(size_t i=0;i<n;i++) {
        if(typeM==1){
          Mzprop[i] = iM[i] - (MOffset+uM[u_index[i]]);
          YMzlik_prop += R::dnorm(Mzprop[i], 0., iMsigest, true);
        } else if(typeM==2){
          Mzprop[i] = Msign[i] * rtnorm(Msign[i]*mBM.f(i), -Msign[i]*(MOffset+uM[u_index[i]]), 1., gen);
          YMzlik_prop += R::pnorm(Msign[i] * (MOffset+mBM.f(i)+uM[u_index[i]]), 0., 1., true, true);
        }
        if(typeY==1){
          Yzprop[i] = iY[i] - (YOffset+uY[u_index[i]]);
          YMzlik_prop += R::dnorm(Yzprop[i], 0., iYsigest, true);
        } else if(typeY==2){
          Yzprop[i] = Ysign[i] * rtnorm(Ysign[i]*yBM.f(i), -Ysign[i]*(YOffset+uY[u_index[i]]), 1., gen);
          YMzlik_prop += R::pnorm(Ysign[i] * (YOffset+yBM.f(i)+uY[u_index[i]]), 0., 1., true, true);
        }
      }
      
      Mz = Mzprop;
      Yz = Yzprop;
      
      // // acceptance ratio
      // ratio = exp(YMzlik_prop-YMzlik);
      // if (ratio > gen.uniform()){
      //   YMzlik = YMzlik_prop;
      //   Mz = Mzprop;
      //   Yz = Yzprop;
      // }
      
      //--------------------------------------------------
      // draw iMsigest and iYsigest
      if(typeM1){
        double Mrss = 0.;
        for(size_t i=0;i<n;i++) {
          Mrss += pow((iM[i]-(MOffset+mBM.f(i)+uM[u_index[i]])), 2.);
        }
        iMsigest = sqrt((nu*Mlambda + Mrss)/gen.chi_square(df));
      }
      if(typeY1){
        double Yrss = 0.;
        for(size_t i=0;i<n;i++) {
          Yrss += pow((iY[i]-(YOffset+yBM.f(i)+uY[u_index[i]])), 2.);
        }
        iYsigest = sqrt((nu*Ylambda + Yrss)/gen.chi_square(df));
      }
      
      // //--------------------------------------------------
      // //--------------------------------------------------
      // // draw uM, uY
      // double mu_uM_prop = gen.normal(), mu_uY_prop = gen.normal();
      // for(size_t j=0; j<J; j++) {
      //   mu_uM_prop += uM[j];
      //   mu_uY_prop += uY[j];
      // }
      // mu_uM_prop /= J;
      // mu_uY_prop /= J;
      // 
      // double sd_uM_prop = 0., sd_uY_prop = 0.;
      // for(size_t j=0; j<J; j++) {
      //   sd_uM_prop += pow(uM[j] - mu_uM_prop, 2.);
      //   sd_uY_prop += pow(uY[j] - mu_uY_prop, 2.);
      // }
      // sd_uM_prop = sqrt(sd_uM_prop/(J-1.));
      // sd_uY_prop = sqrt(sd_uY_prop/(J-1.));
      // 
      // //--------------------------------------------------
      // // draw tau_uM, tau_uY
      // double sum_uM2, sum_uY2;
      // sum_uM2=0.,sum_uY2=0.;
      // for(size_t j=0; j<J; j++) {
      //   sum_uM2 += pow(uM[j] - mu_uM_prop, 2.);
      //   sum_uY2 += pow(uY[j] - mu_uY_prop, 2.);
      // }
      // sd_uM_prop = 1/rtgamma(0.5*(J-1.), 0.5*(sum_uM2), invB2M, gen); sd_uM_prop = std::max(sd_uM_prop, sqrt(B_uM * gen.uniform()));
      // sd_uM_prop = 1/rtgamma(0.5*(J-1.), 0.5*(sum_uY2), invB2Y, gen); sd_uM_prop = std::max(sd_uM_prop, sqrt(B_uY * gen.uniform()));
      // 
      // double rho_uYM_prop = gen.uniform() * 2 - 1; // gen.uniform();
      // // double sd_uM_prop = sqrt(5 * B_uM * gen.uniform());
      // // double sd_uY_prop = sqrt(5 * B_uY * gen.uniform());
      // // double mu_uM_prop = gen.normal() * 5;
      // // double mu_uY_prop = gen.normal() * 5;
      // 
      // //--------------------------------------------------
      // //--------------------------------------------------
      // size_t n_j, ii;
      // double uYMlik_prop = 0.;
      // double *uMprop = new double[J];
      // double *uYprop = new double[J];
      // 
      // ii = 0;
      // for(size_t j=0; j<J; j++) {
      //   n_j = n_j_vec[j];
      //   
      //   double mu_uM_j = mu_uM_prop, sd_uM_j = sd_uM_prop;
      //   double mu_uY_j = mu_uY_prop, sd_uY_j = sd_uY_prop;
      //   
      //   uMprop[j] = gen.normal() * sd_uM_j + mu_uM_j;
      //   mu_uY_j += (sd_uY_j / sd_uM_j) * rho_uYM_prop * (uMprop[j] - mu_uM_j);
      //   sd_uY_j *= sqrt(1 - pow(rho_uYM_prop, 2));
      //   uYprop[j] = gen.normal() * sd_uY_j + mu_uY_j;
      //   
      //   uYMlik_prop +=
      //     R::dnorm(uMprop[j], mu_uM_j, sd_uM_j, true) +
      //     R::dnorm(uYprop[j], mu_uY_j, sd_uY_j, true);
      //   if (typeM==1 && typeY==1) {
      //     for(size_t itmp=0; itmp<n_j; itmp++) {
      //       uYMlik_prop +=
      //         R::dnorm(iM[ii], MOffset + mBM.f(ii) + uMprop[j], iMsigest, true) +
      //         R::dnorm(iY[ii], YOffset + yBM.f(ii) + uYprop[j], iYsigest, true);
      //       ii++;
      //     }
      //   } else if (typeM==2 && typeY==1) {
      //     for(size_t itmp=0; itmp<n_j; itmp++) {
      //       uYMlik_prop +=
      //         R::pnorm(Msign[ii]*(MOffset + mBM.f(ii) + uMprop[j]), 0., 1., true, true) +
      //         R::dnorm(iY[ii], YOffset + yBM.f(ii) + uYprop[j], iYsigest, true);
      //       ii++;
      //     }
      //   } else if (typeM==1 && typeY==2) {
      //     for(size_t itmp=0; itmp<n_j; itmp++) {
      //       uYMlik_prop +=
      //         R::dnorm(iM[ii], MOffset + mBM.f(ii) + uMprop[j], iMsigest, true) +
      //         R::pnorm(Ysign[ii]*(YOffset + yBM.f(ii) + uYprop[j]), 0., 1., true, true);
      //       ii++;
      //     }
      //   } else if (typeM==2 && typeY==2) {
      //     for(size_t itmp=0; itmp<n_j; itmp++) {
      //       uYMlik_prop +=
      //         R::pnorm(Msign[ii]*(MOffset + mBM.f(ii) + uMprop[j]), 0., 1., true, true) +
      //         R::pnorm(Ysign[ii]*(YOffset + yBM.f(ii) + uYprop[j]), 0., 1., true, true);
      //       ii++;
      //     }
      //   }
      // }
      // 
      // // acceptance ratio
      // ratio = exp(uYMlik_prop-uYMlik);
      // if (ratio > gen.uniform()){
      //   uYMlik = uYMlik_prop;
      //   mu_uM = mu_uM_prop;
      //   mu_uY = mu_uY_prop;
      //   sd_uM = sd_uM_prop;
      //   sd_uY = sd_uY_prop;
      //   rho_uYM = rho_uYM_prop;
      //   uM = uMprop;
      //   uY = uYprop;
      // }
      
      //--------------------------------------------------
      //--------------------------------------------------
      // draw uM, uY
      double mu_uM_prop = 0., mu_uY_prop = 0.;
      for(size_t j=0; j<J; j++) {
        mu_uM_prop += uM[j];
        mu_uY_prop += uY[j];
      }
      mu_uM_prop /= J;
      mu_uY_prop /= J;

      //--------------------------------------------------
      // draw tau_uM, tau_uY
      double sd_uM_prop = 0., sd_uY_prop = 0.;
      for(size_t j=0; j<J; j++) {
        sd_uM_prop += pow(uM[j] - mu_uM_prop, 2.);
        sd_uY_prop += pow(uY[j] - mu_uY_prop, 2.);
      }
      sd_uM_prop = sqrt(1/rtgamma(0.5*(J-1.), 0.5*sd_uM_prop, invB2M, gen));
      sd_uY_prop = sqrt(1/rtgamma(0.5*(J-1.), 0.5*sd_uY_prop, invB2Y, gen));
      sd_uM_prop = std::max(sd_uM_prop, 1/sqrt(B_uM * gen.uniform()));
      sd_uY_prop = std::max(sd_uY_prop, 1/sqrt(B_uY * gen.uniform()));
      
      //--------------------------------------------------
      //--------------------------------------------------
      // draw uM, uY
      size_t n_j, ii, ii_j;
      double rho_uYM_prop;
      
      double *uYMlik_j_prop = new double[J];
      double *uMprop = new double[J];
      double *uYprop = new double[J];
      
      ii=0;
      rho_uYM_prop = gen.uniform() * 2 - 1; // gen.uniform();
      //--------------------------------------------------
      //--------------------------------------------------
      for(size_t j=0; j<J; j++) {
        n_j = n_j_vec[j];
        
        double mu_uM_j = mu_uM_prop, sd_uM_j = sd_uM_prop;
        double mu_uY_j = mu_uY_prop, sd_uY_j = sd_uY_prop;
        
        uMprop[j] = gen.normal() * sd_uM_j + mu_uM_j;
        mu_uY_j += (sd_uY_j / sd_uM_j) * rho_uYM_prop * (uMprop[j] - mu_uM_j);
        sd_uY_j *= sqrt(1 - pow(rho_uYM_prop, 2));
        uYprop[j] = gen.normal() * sd_uY_j + mu_uY_j;
        
        uYMlik_j_prop[j] =
          R::dnorm(uMprop[j], mu_uM_j, sd_uM_j, true) +
          R::dnorm(uYprop[j], mu_uY_j, sd_uY_j, true);
        if (typeM==1 && typeY==1) {
          for(size_t itmp=0; itmp<n_j; itmp++) {
            uYMlik_j_prop[j] +=
              R::dnorm(iM[ii], MOffset + mBM.f(ii) + uMprop[j], iMsigest, true) +
              R::dnorm(iY[ii], YOffset + yBM.f(ii) + uYprop[j], iYsigest, true);
            ii++;
          }
        } else if (typeM==2 && typeY==1) {
          for(size_t itmp=0; itmp<n_j; itmp++) {
            uYMlik_j_prop[j] +=
              R::pnorm(Msign[ii]*(MOffset + mBM.f(ii) + uMprop[j]), 0., 1., true, true) +
              R::dnorm(iY[ii], YOffset + yBM.f(ii) + uYprop[j], iYsigest, true);
            ii++;
          }
        } else if (typeM==1 && typeY==2) {
          for(size_t itmp=0; itmp<n_j; itmp++) {
            uYMlik_j_prop[j] +=
              R::dnorm(iM[ii], MOffset + mBM.f(ii) + uMprop[j], iMsigest, true) +
              R::pnorm(Ysign[ii]*(YOffset + yBM.f(ii) + uYprop[j]), 0., 1., true, true);
            ii++;
          }
        } else if (typeM==2 && typeY==2) {
          for(size_t itmp=0; itmp<n_j; itmp++) {
            uYMlik_j_prop[j] +=
              R::pnorm(Msign[ii]*(MOffset + mBM.f(ii) + uMprop[j]), 0., 1., true, true) +
              R::pnorm(Ysign[ii]*(YOffset + yBM.f(ii) + uYprop[j]), 0., 1., true, true);
            ii++;
          }
        }
        
        // acceptance ratio
        ratio = exp(uYMlik_j_prop[j]-uYMlik_j[j]);
        if (ratio > gen.uniform()){
          uYMlik_j[j] = uYMlik_j_prop[j];
          uM[j] = uMprop[j];
          uY[j] = uYprop[j];
        }
      }
      
      // acceptance ratio
      ratio = 0.;
      for(size_t j=0; j<J; j++) {
        ratio += uYMlik_j_prop[j] - uYMlik_j[j];
      }
      if (exp(ratio) > gen.uniform()){
        mu_uM = mu_uM_prop;
        mu_uY = mu_uY_prop;
        sd_uM = sd_uM_prop;
        sd_uY = sd_uY_prop;
        rho_uYM = rho_uYM_prop;
      }
      
      //--------------------------------------------------
      if(postrep>=burn) {
        if(postrep%printevery==0) {
          printf("done %zu (out of %zu)\n",postrep,numdraw+burn);
        }
        if(nkeeptrain && (((postrep-burn+1) % skiptr) == 0)) {
          for(size_t i=0;i<n;i++) {
            MDRAW(trcnt,i) = MOffset + mBM.f(i);
            YDRAW(trcnt,i) = YOffset + yBM.f(i);
            // MDRAW(trcnt,i) = MOffset + mBM.f(i) + uM[u_index[i]];
            // YDRAW(trcnt,i) = YOffset + yBM.f(i) + uY[u_index[i]];
          }
          Msdraw[trcnt]=iMsigest;
          Ysdraw[trcnt]=iYsigest;
          
          for(size_t j=0;j<J;j++) {
            UMDRAW(trcnt,j) = uM[j];
            UYDRAW(trcnt,j) = uY[j];
          }
          rhoYMudraw[trcnt] = rho_uYM;
          sdMudraw[trcnt] = sd_uM;
          sdYudraw[trcnt] = sd_uY;
          muMudraw[trcnt] = mu_uM;
          muYudraw[trcnt] = mu_uY;
          
          trcnt+=1;
          
          keeptreedraw = nkeeptreedraws && (((postrep-burn+1) % skiptreedraws) == 0);
          if(keeptreedraw) {
            matXtreess << ",";
            matMtreess << ",";
            for(size_t j=0;j<numtree;j++) {
              matXtreess << mBM.gettree(j);
              matMtreess << yBM.gettree(j);
#ifndef NoRcpp
              imatXvarcnt=mBM.getnv();
              imatXvarprb=mBM.getpv();
              size_t it=(postrep-burn)/skiptreedraws;
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
    ret["cor.uYM"]=rhoYMudraw;
    ret["sd.uM"]=sdMudraw;
    ret["sd.uY"]=sdYudraw;
    ret["mu.uM"]=muMudraw;
    ret["mu.uY"]=muYudraw;
    ret["matXvarcount"]=matXvarcnt;
    ret["matMvarcount"]=matMvarcnt;
    ret["matXvarprob"]=matXvarprb;
    ret["matMvarprob"]=matMvarprb;
    
    Rcpp::List matxiret(matXxi.size());
    for(size_t it=0;it<matXxi.size();it++) {
      Rcpp::NumericVector vtemp(matXxi[it].size());
      std::copy(matXxi[it].begin(),matXxi[it].end(),vtemp.begin());
      matxiret[it] = Rcpp::NumericVector(vtemp);
    }
    Rcpp::List matmiret(matMxi.size());
    for(size_t it=0;it<matMxi.size();it++) {
      Rcpp::NumericVector vtemp(matMxi[it].size());
      std::copy(matMxi[it].begin(),matMxi[it].end(),vtemp.begin());
      matmiret[it] = Rcpp::NumericVector(vtemp);
    }
    
    Rcpp::List matXtreesL;
    matXtreesL["cutpoints"] = matxiret;
    matXtreesL["trees"]=Rcpp::CharacterVector(matXtreess.str());
    ret["matXtreedraws"] = matXtreesL;
    Rcpp::List matMtreesL;
    matMtreesL["cutpoints"] = matmiret;
    matMtreesL["trees"]=Rcpp::CharacterVector(matMtreess.str());
    ret["matMtreedraws"] = matMtreesL;
    
    return ret;
#else
    
#endif
    
  }