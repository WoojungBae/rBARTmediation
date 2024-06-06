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
#include "heterbart.h"
#include "rtnorm.h"
#include "rtgamma.h"
#include "lambda.h"

#ifndef NoRcpp

#define YDRAW(a, b) ydraw(a, b)
#define UDRAW(a, b) udraw(a, b)

RcppExport SEXP crBART(SEXP _typeY,   // 1:continuous, 2:binary, 3:multinomial
                       SEXP _in,      // number of observations in training data
                       SEXP _ip,      // dimension of x
                       SEXP _imatX,   // x, train, p x n  // (transposed so rows are contiguous in memory)
                       SEXP _iY,      // y, train, n x 1
                       SEXP _iu_index,
                       SEXP _in_j_vec,
                       SEXP _iu,
                       SEXP _iJ,
                       SEXP _iB,
                       SEXP _inumtree,  // number of trees
                       SEXP _inumcut,   // number of cut points
                       SEXP _inumdraw,  // number of kept draws (except thinnning)
                       SEXP _iburn,     // number of burn-in draws skipped
                       SEXP _ithin,     // thinning
                       SEXP _ipower,
                       SEXP _ibase,
                       SEXP _Offset,
                       SEXP _itau,
                       SEXP _inu,
                       SEXP _ilambda,
                       SEXP _isigest,
                       SEXP _idart,   // dart prior: true(1)=yes, false(0)=no
                       SEXP _itheta,
                       SEXP _iomega,
                       SEXP _ia,      // param a for sparsity prior
                       SEXP _ib,      // param b for sparsity prior
                       SEXP _irho,    // param rho for sparsity prior (default=p)
                       SEXP _iaug,    // category strategy:  true(1)=data augment 
                       SEXP _inprintevery,
                       SEXP _Xinfo){
  // process args
  int typeY = Rcpp::as<int>(_typeY);
  size_t n = Rcpp::as<int>(_in);
  size_t p = Rcpp::as<int>(_ip);
  Rcpp::NumericVector  matXv(_imatX);
  double *imatX = &matXv[0];
  Rcpp::NumericVector  Yv(_iY); 
  double *iY = &Yv[0];
  Rcpp::IntegerVector _u_index(_iu_index);
  int *u_index = &_u_index[0];
  Rcpp::IntegerVector _n_j_vec(_in_j_vec);
  int *n_j_vec = &_n_j_vec[0];
  Rcpp::NumericVector _u(_iu);
  double *u = &_u[0];
  size_t J = Rcpp::as<int>(_iJ);
  double B = Rcpp::as<double>(_iB);
  size_t numtree = Rcpp::as<int>(_inumtree);
  Rcpp::IntegerVector _nc(_inumcut);
  int *numcut = &_nc[0];
  size_t numdraw = Rcpp::as<int>(_inumdraw);
  size_t burn = Rcpp::as<int>(_iburn);
  size_t thin = Rcpp::as<int>(_ithin);
  double mybeta = Rcpp::as<double>(_ipower);
  double alpha = Rcpp::as<double>(_ibase);
  double Offset = Rcpp::as<double>(_Offset);
  double tau = Rcpp::as<double>(_itau);
  double nu = Rcpp::as<double>(_inu);
  double lambda = Rcpp::as<double>(_ilambda);
  double sigma=Rcpp::as<double>(_isigest);
  bool dart;
  if(Rcpp::as<int>(_idart)==1) dart=true;
  else dart=false;
  double a = Rcpp::as<double>(_ia);
  double b = Rcpp::as<double>(_ib);
  double rho = Rcpp::as<double>(_irho);
  bool aug;
  if(Rcpp::as<int>(_iaug)==1) aug=true;
  else aug=false;
  double theta = Rcpp::as<double>(_itheta);
  double omega = Rcpp::as<double>(_iomega);
  size_t nkeeptrain = numdraw/thin;     // Rcpp::as<int>(_inkeeptrain);
  size_t nkeeptreedraws = numdraw/thin; // Rcpp::as<int>(_inkeeptreedraws);
  size_t printevery = Rcpp::as<int>(_inprintevery);
  Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
  Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
  Rcpp::NumericMatrix Xinfo(_Xinfo);
  Rcpp::NumericVector sdraw(numdraw+burn);
  Rcpp::NumericVector sdudraw(numdraw+burn);
  Rcpp::NumericMatrix ydraw(nkeeptrain,n);
  Rcpp::NumericMatrix udraw(nkeeptrain,J);
  
  //random number generation
  arn gen;
  
  // heterbart bm(numtree);
  bart bm(numtree);
  
  if(Xinfo.size()>0) {
    xinfo _xi;
    _xi.resize(p);
    for(size_t q=0;q<p;q++) {
      _xi[q].resize(numcut[q]);
      for(size_t j=0;j<numcut[q];j++) _xi[q][j]=Xinfo(q, j);
    }
    bm.setxinfo(_xi);
  }
#else
  
#define YDRAW(a, b) ydraw[a][b]
#define UDRAW(a, b) udraw[a][b]
  
  void crBART(int typeY,     // 1:continuous, 2:binary, 3:multinomial
              size_t n,      // number of observations in training data
              size_t p,	     // dimension of x
              double* imatX, // x, train, p x n //(transposed so rows are contiguous in memory)
              double* iY,    // y, train, n x 1
              int *u_index,
              int *n_j_vec,
              double *u,
              size_t J,
              double B,
              size_t numtree,	// number of trees
              int *numcut,    // number of cut points
              size_t numdraw,	// number of kept draws (except for thinnning ..)
              size_t burn,    // number of burn-in draws skipped
              size_t thin,    // thinning
              double mybeta,
              double alpha,
              double Offset,
              double tau,
              double nu,
              double lambda,
              double sigma,
              bool dart,        // dart prior: true(1)=yes, false(0)=no   
              double theta,
              double omega, 
              double a,	        // param a for sparsity prior                         
              double b,	        // param b for sparsity prior                        
              double rho,       // param rho for sparsity prior (default to p)   
              bool aug,         // categorical strategy: true(1)=data augment  
              size_t printevery,
              unsigned int n1,  // additional parameters needed to call from C++
              unsigned int n2,
              double* sdraw,
              double* sdudraw,
              double* _ydraw,
              double* _udraw) {
    //--------------------------------------------------
    //return data structures (using C++)
    size_t nkeeptrain=numdraw/thin, nkeeptreedraws=numdraw/thin;
    std::vector<double*> ydraw(nkeeptrain);
    std::vector<double*> udraw(nkeeptrain);
    
    for(size_t it=0; it<nkeeptrain; ++it) {
      ydraw[it]=&_ydraw[it*n];
      udraw[it]=&_udraw[it*J];
    }
    
    //matrix to return dart posteriors (counts and probs)
    std::vector< std::vector<size_t> > varcnt;
    std::vector< std::vector<double> > varprb;
    
    //random number generation
    arn gen(n1, n2);
    
    //heterbart bm(numtree);
    bart bm(numtree);
#endif
    
    std::stringstream treess;  //string stream to write trees to
    treess.precision(10);
    treess << nkeeptreedraws << " " << numtree << " " << p << endl;
    
    printf("*****Calling rBART: typeY=%d\n", typeY);
    
    size_t skiptr=thin, skiptreedraws=thin;
    
    //--------------------------------------------------
    // print args
    printf("*****Data:\n");
    printf("data:n,p: %zu, %zu\n",n,p);
    printf("y1,yn: %lf, %lf\n",iY[0],iY[n-1]);
    printf("x1,x[n*p]: %lf, %lf\n",imatX[0],imatX[n*p-1]);
    //   if(hotdeck) 
    //printf("warning: missing elements in x multiply imputed with hot decking\n");
    printf("*****Number of Trees: %zu\n",numtree);
    printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
    printf("*****burn,numdraw,thin: %zu,%zu,%zu\n",burn,numdraw,thin);
    // printf("Prior:\nbeta,alpha,tau,nu,lambda,offset: %lf,%lf,%lf,%lf,%lf,%lf\n",
    //                    mybeta,alpha,tau,nu,lambda,Offset);
    cout << "*****Prior:beta,alpha,tau,nu,lambda,offset: " 
         << mybeta << ',' << alpha << ',' << tau << ',' 
         << nu << ',' << lambda << ',' << Offset << endl;
    if(typeY==1) {
      printf("*****sigma: %lf\n",sigma);
    }
    cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: " 
         << dart << ',' << theta << ',' << omega << ',' << a << ',' 
         << b << ',' << rho << ',' << aug << endl;
    printf("*****printevery: %zu\n",printevery);
    
    //--------------------------------------------------
    //create temporaries
    double df = n + nu;
    double *z = new double[n]; 
    // double *svec = new double[n]; 
    double *sign;
    if(typeY!=1) sign = new double[n]; 
    for(size_t i=0; i<n; i++) {
      if(typeY==1) {
        // svec[i] = 1.;
        // svec[i] = sigma; 
        z[i] = iY[i]-Offset; 
      } else {
        if(iY[i]==0) {
          sign[i] = -1.;
        } else {
          sign[i] = 1.;
        }
        z[i] = sign[i];
      }
    }
    
    // double *u = new double[J]; 
    double invB2=pow(B, -2.), sd_u=B*0.5, tau_u=pow(sd_u, -2.); // invB2_16=16.*invB2, 
    
    if(u[0]!=u[0]) {
      for(size_t j=0; j<J; j++) {
        u[j]=sd_u*gen.normal();
      }
    }
    
    // set up BART model
    bm.setprior(alpha,mybeta,tau);
    bm.setdata(p,n,imatX,z,numcut);
    bm.setdart(a,b,rho,aug,dart);
    
    // dart iterations
    std::vector<double> ivarprb (p,0.);
    std::vector<size_t> ivarcnt (p,0);
    
    //--------------------------------------------------
    // mcmc
    printf("\nMCMC\n");
    // size_t index;
    size_t trcnt=0; // count kept train draws
    bool keeptreedraw, type1sigest=(typeY==1 && lambda!=0.);
    
    time_t tp;
    int time1 = time(&tp), total=numdraw+burn;
    xinfo& xi = bm.getxinfo();
    
    for(size_t postrep=0;postrep<total;postrep++) {
      if(postrep%printevery==0) printf("done %zu (out of %lu)\n",postrep,numdraw+burn);
      if(postrep==(burn/2)&&dart) bm.startdart();
      //--------------------------------------------------
      // draw bart
      // bm.draw(svec,gen);
      bm.draw(sigma,gen);
      
      //--------------------------------------------------
      for(size_t i=0;i<n;i++) {
        if(typeY==1){
          z[i] = iY[i] - (Offset+u[u_index[i]]);
          // svec[i] = 1.*sigma; 
        } else if(typeY==2){
          z[i] = sign[i] * rtnorm(sign[i]*bm.f(i), -sign[i]*(Offset+u[u_index[i]]), sigma, gen);
          // svec[i]=sqrt(draw_lambda_i(pow(svec[i], 2.), sign[i]*(bm.f(i)+Offset+u[u_index[i]]), 1000, 1, gen));
        } else if(typeY==3){
          z[i] = sign[i] * rtnorm(sign[i]*bm.f(i), -sign[i]*(Offset+u[u_index[i]]), sigma, gen);
          // svec[i]=sqrt(draw_lambda_i(pow(svec[i], 2.), sign[i]*(bm.f(i)+Offset+u[u_index[i]]), 1000, 1, gen));
        }
      }
      
      //--------------------------------------------------
      // draw sigma
      if(type1sigest) {
        double rss=0.;
        for(size_t i=0;i<n;i++) {
          rss += pow((iY[i]-bm.f(i)-(Offset+u[u_index[i]])), 2.); 
        }
        sigma = sqrt((nu*lambda + rss)/gen.chi_square(df));
        sdraw[postrep]=sigma;
      }
      
      //--------------------------------------------------
      // draw tau_u
      double sum_u2;
      sum_u2=0.;
      for(size_t j=0; j<J; j++) {
        sum_u2 += pow(u[j], 2.);
      }
      tau_u = rtgamma(0.5*(J-1.), 0.5*sum_u2, invB2, gen); 
      
      //--------------------------------------------------
      // draw u
      size_t jj, n_j;
      double mu_u_j, sd_u_j, prec=pow(sigma, -2.);
      
      jj=0;
      for(size_t j=0; j<J; j++) {
        n_j=n_j_vec[j];
        mu_u_j=0.;
        sd_u_j=pow(tau_u+n_j*prec, -0.5);
        for(size_t j=0; j<n_j; j++) {
          mu_u_j += (iY[jj]-bm.f(jj)-Offset);
          jj++;
        }
        mu_u_j *= prec*pow(sd_u_j, 2.);
        u[j]=gen.normal()*sd_u_j+mu_u_j;
      }
      
      sdudraw[postrep]=pow(tau_u, -0.5);
      
      //--------------------------------------------------
      if(postrep>=burn) {
        if(nkeeptrain && (((postrep-burn+1) % skiptr) ==0)) {
          for(size_t i=0;i<n;i++) {
            // YDRAW(trcnt,i)=bm.f(i);
            // YDRAW(trcnt,i)=Offset+bm.f(i);
            YDRAW(trcnt,i)=Offset+bm.f(i)+u[u_index[i]];
          }
          for(size_t j=0;j<J;j++) {
            UDRAW(trcnt,j)=u[j];
          }
          trcnt+=1;
        }
        keeptreedraw = nkeeptreedraws && (((postrep-burn+1) % skiptreedraws) ==0);
        if(keeptreedraw) {
          for(size_t j=0;j<numtree;j++) {
            treess << bm.gettree(j);
            
#ifndef NoRcpp
            ivarcnt=bm.getnv();
            ivarprb=bm.getpv();
            size_t q=(postrep-burn)/skiptreedraws;
            for(size_t j=0;j<p;j++){
              varcnt(q,j)=ivarcnt[j];
              varprb(q,j)=ivarprb[j];
            }
#else
            varcnt.push_back(bm.getnv());
            varprb.push_back(bm.getpv());
#endif
          }
        }
      }
    }
    int time2 = time(&tp);
    printf("time: %ds\n",time2-time1);
    printf("trcnt: %zu\n",trcnt);
    
    delete[] z;
    // delete[] svec;
    // delete[] u;
    if(typeY!=1) delete[] sign;
    
#ifndef NoRcpp
    
    //--------------------------------------------------
    //return list
    Rcpp::List ret;
    if(type1sigest) ret["sigma"]=sdraw;
    ret["y.draw"]=ydraw;
    ret["varcount"]=varcnt;
    ret["varprob"]=varprb;
    ret["u.draw"]=udraw;
    ret["sd.u"]=sdudraw;
    
    Rcpp::List xiret(xi.size());
    for(size_t it=0;it<xi.size();it++) {
      Rcpp::NumericVector vtemp(xi[it].size());
      std::copy(xi[it].begin(),xi[it].end(),vtemp.begin());
      xiret[it] = Rcpp::NumericVector(vtemp);
    }
    
    Rcpp::List treesL;
    treesL["cutpoints"] = xiret;
    treesL["trees"]=Rcpp::CharacterVector(treess.str());
    ret["treedraws"] = treesL;
    
    return ret;
#else
    
#endif
    
  }
  