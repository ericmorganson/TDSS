#!/usr/bin/env python

import os
import lsd
import lsd.colgroup as colgroup
import lsd.bounds
import numpy as np
import scipy.stats.mstats
from itertools import izip
from collections import defaultdict
from lsd.join_ops import IntoWriter
from scipy.weave import inline
import logging
import cPickle
import calib as cal
import pdb
import pyfits
import numpy as np, matplotlib,os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import cosmolopy.distance as cd

cosmo={'omega_M_0' : 0.27, 'omega_lambda_0' : 0.7, 'h' : 0.72, 'omega_k_0' : 0 }

#def mu(z):
#  return 5.0*np.log10(cd.luminosity_distance(z=z, **cosmo)*10**5)

def redo(gamma=0.55):
  a1=letsdothisthang(num=0)
  a2=letsdothisthang(num=1)
  a3=letsdothisthang(num=2)
  a4=letsdothisthang(num=3)
#  print a1[1], a2[1], a3[1], a4[1]
  return gamma, a1[0], a2[0], a3[0], a4[0], a1[0]+a2[0]+a3[0]+a4[0]

def fitline(xfit,yfit,wtfit):
  S=np.sum(wtfit); Sx=np.sum(xfit*wtfit); Sy=np.sum(yfit*wtfit); Sxx=np.sum(xfit**2*wtfit); Sxy=np.sum(xfit*yfit*wtfit)
  Delta=S*Sxx-Sx**2; a=(Sxx*Sy-Sx*Sxy)/Delta; b=(S*Sxy-Sx*Sy)/Delta; siga=np.sqrt(Sxx/Delta); sigb=np.sqrt(S/Delta)
  chi2=np.sum((yfit-a-b*xfit)**2*wtfit)/(xfit.size-2.0)
  out=np.zeros(6)
  out[0]=np.exp(a)
  out[1]=b
  out[2]=np.exp(a)*siga
  out[3]=sigb
  out[4]=-Sx/Delta*np.exp(a)
  out[5]=chi2
  return out 

def makeplot(var,var_wt,var_t,am,num,ext=''):
  filters=['g','r','i','z']
  lstring=filters[num]+'$_{\mathrm{P1}}$'
  outname="ps1_sdss_qso_"+str(num)+ext+".png"
  nqsos,npoints=var.shape
  data1d=np.sum(var*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  data21d=np.sum(var*var*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  std1d=np.sqrt((data21d-data1d**2)/(np.sum(var_wt>0,axis=0)-.99))
  std1d= np.std(var,axis=0)/np.sqrt(-.99+np.sum(var_wt>0,axis=0))
  var=np.std(var[var_t>1])
  std1d2=np.sqrt(var/(np.sum(var_wt>0,axis=0)+.0001))
  std1d=np.max(np.vstack([std1d,std1d2]),axis=0)
  t1d=np.sum(var_t*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  scale1d=t1d
  time=(t1d**(1.0/am[1]))[data1d>0]
  std1d=(std1d*scale1d)[data1d > 0]
  scale1d=scale1d[data1d > 0]
  t1d=t1d[data1d>0]
  data1d=(data1d[data1d > 0]*scale1d)
  chi2=np.sum((data1d-am[0]*scale1d)**2/std1d**2)/(np.sum(std1d>0)-2)
  xfit=np.log(time); yfit=np.log(data1d); wtfit=(data1d/std1d)**2
# Switch to V(t) rather that V^2(t)
  data1d=np.sqrt(data1d); std1d=std1d*0.5/data1d
  ylower = np.maximum(1e-10, data1d - std1d)
  yerr_lower = data1d - ylower
  yerr=[yerr_lower, std1d]
  xplt=10**np.arange(-3,01.1,.01)
  yplt=np.sqrt(am[0]*xplt**am[1])
  plt.figure(figsize=(8,7.2))
  plt.subplot(111, xscale="log", yscale="log")
  plt.rcParams['font.size'] = 20
  plt.xlim((.005,12))
  plt.ylim((.01,1))
  plt.errorbar(time,data1d,yerr=yerr,fmt='ok')  
  plt.plot(xplt,yplt,'k-')
  plt.text(0.01, 0.5, lstring, size = 36)
  plt.xlabel('Time (years)')
  plt.ylabel('V(t) (mag)')
  plt.savefig(outname)
  plt.clf()
  S=np.sum(wtfit); Sx=np.sum(xfit*wtfit); Sy=np.sum(yfit*wtfit); Sxx=np.sum(xfit**2*wtfit); Sxy=np.sum(xfit*yfit*wtfit)
  Delta=S*Sxx-Sx**2; a=(Sxx*Sy-Sx*Sxy)/Delta; b=(S*Sxy-Sx*Sy)/Delta; siga=np.sqrt(Sxx/Delta); sigb=np.sqrt(S/Delta)
#  print str(np.exp(a))+' +/- '+str(np.exp(a)*siga)+', '+str(b)+' +/- '+str(sigb)+', cov = '+str(-Sx/Delta*np..exp(a))
  out=np.zeros(6)
  out[0]=np.exp(a)
  out[1]=b
  out[2]=np.exp(a)*siga
  out[3]=sigb
  out[4]=-Sx/Delta*np.exp(a)
  out[5]=chi2
  return out 

def drw_chi2(p,t,A,sig,cnum,cp):
  if cnum < 4:
    p[cnum]=cp
#  dAs=A-p[0]*(1.0-np.exp(-t/p[1]))**0.5
  dAs=A-np.sqrt((p[0]*t**p[2]/(1+(t/p[1])**p[2]))**2+p[3]**2)
  return np.sum((dAs[sig>0])**2/(sig[sig>0])**2)

chi22 = lambda p, t, A, sig, cnum, cp: drw_chi2(p,t,A,sig,cnum,cp)

def sigma(p,t,A,sig,cnum):
  delta=0.1*abs(p[cnum])
  y0=scipy.optimize.fmin(chi22,p,args=[t,A,sig,cnum,p[cnum]-delta],disp=0,full_output=1,ftol=.000001)[1]
  y1=scipy.optimize.fmin(chi22,p,args=[t,A,sig,cnum,p[cnum]],disp=0,full_output=1,ftol=.000001)[1]
  y2=scipy.optimize.fmin(chi22,p,args=[t,A,sig,cnum,p[cnum]+delta],disp=0,full_output=1,ftol=.000001)[1]
  return delta*np.sqrt(2.0/(y2+y0-2.0*y1))

def cov(p,t,A,sig):
  delta0=0.1*abs(p[0])
  delta1=0.1*abs(p[1])
  y0=drw_chi2(p,t,A,sig,2,0)
  y1=drw_chi2([p[0]+delta0,p[1]+delta1],t,A,sig,2,0)
  y2=drw_chi2([p[0]-delta0,p[1]-delta1],t,A,sig,2,0)
  y3=drw_chi2([p[0]+delta0,p[1]-delta1],t,A,sig,2,0)
  y4=drw_chi2([p[0]-delta0,p[1]+delta1],t,A,sig,2,0)
  return 4*delta0*delta1/(y1+y2+y3+y4) 

def fitfunc(t,A,sig):
  p0=[0.4,2,0.5,0.04]
  p1,rchi2=scipy.optimize.fmin(chi22,p0,args=[t,A,sig,4,0],disp=1,full_output=1,ftol=.000001,maxiter=1000)[:2]
  psig=np.zeros(4)
  rchi2=rchi2/(np.sum(sig>0)-3.999)
  for num in range(4):
    psig[num]=sigma(p1,t,A,sig,num)
#  covar=cov(p1,t,A,sig)
  return [p1[0], p1[1], p1[2], p1[3], psig[0], psig[1],psig[2], psig[3], rchi2]



def makeplot2(var,var_wt,var_t,am,num,ext=''):
  filters=['g','r','i','z']
  lstring=filters[num]+'$_{\mathrm{P1}}$'
  outname="ps1_sdss_qso_"+str(num)+ext+"_drw.png"
  nqsos,npoints=var.shape
  data1d=np.sum(var*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  data21d=np.sum(var*var*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  std1d=np.sqrt((data21d-data1d**2)/(np.sum(var_wt>0,axis=0)-.99))
  std1d= np.std(var,axis=0)/np.sqrt(-.99+np.sum(var_wt>0,axis=0))
  var=np.std(var[var_t>1])
  std1d2=np.sqrt(var/(np.sum(var_wt>0,axis=0)+.0001))
  std1d=np.max(np.vstack([std1d,std1d2]),axis=0)
  t1d=np.sum(var_t*var_wt,axis=0)/(.000001+np.sum(var_wt,axis=0))
  scale1d=t1d
  time=(t1d**(1.0/am[1]))[data1d>0]
  std1d=(std1d*scale1d)[data1d > 0]
  scale1d=scale1d[data1d > 0]
  t1d=t1d[data1d>0]
  data1d=(data1d[data1d > 0]*scale1d)
  chi2=np.sum((data1d-am[0]*scale1d)**2/std1d**2)/(np.sum(std1d>0)-2)
  xfit=np.log(time); yfit=np.log(data1d); wtfit=(data1d/std1d)**2
# Switch to V(t) rather that V^2(t)
  data1d=np.sqrt(data1d); std1d=std1d*0.5/data1d
  out=fitfunc(time,data1d,std1d)
  ylower = np.maximum(1e-10, data1d - std1d)
  yerr_lower = data1d - ylower
  yerr=[yerr_lower, std1d]
  xplt=10**np.arange(-3,01.1,.01)
#  yplt=out[0]*(1-np.exp(-xplt/out[1]))**0.5
  yplt=np.sqrt((out[0]*(xplt)**out[2]/(1+(xplt/out[1])**out[2]))**2+out[3]**2)
  plt.figure(figsize=(8,7.2))
  plt.subplot(111, xscale="log", yscale="log")
  plt.rcParams['font.size'] = 20
  plt.xlim((.005,12))
  plt.ylim((.01,1))
  plt.errorbar(time,data1d,yerr=yerr,fmt='ok')  
  plt.plot(xplt,yplt,'k-')
  plt.text(0.01, 0.5, lstring, size = 36)
  plt.xlabel('Time (years)')
  plt.ylabel('V(t) (mag)')
  plt.savefig(outname)
  plt.clf()
  S=np.sum(wtfit); Sx=np.sum(xfit*wtfit); Sy=np.sum(yfit*wtfit); Sxx=np.sum(xfit**2*wtfit); Sxy=np.sum(xfit*yfit*wtfit)
  Delta=S*Sxx-Sx**2; a=(Sxx*Sy-Sx*Sxy)/Delta; b=(S*Sxy-Sx*Sy)/Delta; siga=np.sqrt(Sxx/Delta); sigb=np.sqrt(S/Delta)
#  print str(np.exp(a))+' +/- '+str(np.exp(a)*siga)+', '+str(b)+' +/- '+str(sigb)+', cov = '+str(-Sx/Delta*np..exp(a))

  outname="ps1_sdss_qso_"+str(num)+ext+"_diff.png"
  plt.figure(figsize=(8,7.2))
  plt.gcf().subplots_adjust(left=0.18)
  plt.subplot(111, xscale="log")
  plt.rcParams['font.size'] = 20
  plt.xlim((.005,12))
  plt.ylim((-0.1,0.1))
  plt.errorbar(time,data1d-np.sqrt(am[0]*time**am[1]),yerr=std1d,fmt='ok')  
  plt.plot(xplt,yplt-np.sqrt(am[0]*xplt**am[1]),'k-')
  plt.axhline(color='black',ls='--')
  plt.text(0.01, 0.08, lstring, size = 36)
  plt.xlabel('Time (years)')
  plt.ylabel('V(t) - A t$^\gamma$ (mag)')
  plt.savefig(outname)
  plt.clf()

  return out 

def drfit(numin=0):
  blah=pyfits.open("qso_sdss_ps1.fits")
#  blah=pyfits.open("/a41217d5/LSD/PS1/ps1_sdss_ps1_160_0_180_5.fits")
  out=blah[1].data
  good= ( out['sdss_ps1'][:,0]<20) &( out['sdss_ps1'][:,0]>0) & ( out['mean'][:,0]-out['mean_ap'][:,0]<0.2) & ( out['mean'][:,1]-out['mean_ap'][:,1]<0.2) & ( out['mean'][:,2]-out['mean_ap'][:,2]<0.2) & ( out['mean'][:,3]-out['mean_ap'][:,2]<0.3) & ( out['sdss_type']==6) & (out['lc_mag'][:,numin,0]>0) & (np.isnan(out['lc_mag'][:,numin,0]) != True) & (np.isnan(out['sdss'][:,numin+1] ) != True) & ( out['sdss'][:,numin+1] > 0 ) & (np.isnan(out['sdss'][:,1] ) != True) & (np.isnan(out['sdss'][:,3] ) != True)
  out=out[good]
  b=[1.4e-10,0.9e-10,1.2e-10,1.8e-10,7.4e-10]
  sdsslim=np.array([23.6,23.3,22.85,21.3])
  for num in range(4):
    out['sdss_ps1'][:,num]=-2.5*np.log10(2.0*b[num+1]*np.sinh(-0.4*np.log(10)*out['sdss'][:,num+1]-np.log(b[num+1])))*(out['sdss'][:,num+1] > 0)
  good= ( out['sdss_ps1'][:,0]<20) &( out['sdss_ps1'][:,0]>0) & ( out['mean'][:,0]-out['mean_ap'][:,0]<0.3) & ( out['mean'][:,1]-out['mean_ap'][:,1]<0.3) & ( out['mean'][:,2]-out['mean_ap'][:,2]<0.3) & ( out['mean'][:,3]-out['mean_ap'][:,3]<0.3) & ( out['sdss_type']==6) & (out['lc_mag'][:,numin,0]>0) & (np.isnan(out['lc_mag'][:,numin,0]) != True) & (np.isnan(out['sdss_ps1'][:,numin] ) != True) & ( out['sdss_ps1'][:,numin] > 0 ) & (np.isnan(out['sdss_ps1'][:,0] ) != True) & (np.isnan(out['sdss_ps1'][:,2] ) != True)
  out=out[good]
  gi=(out['sdss_ps1'][:,0]-out['sdss_ps1'][:,2])*(out['sdss'][:,1] > 0 )*(out['sdss'][:,3] > 0 )*(out['sdss'][:,1] < sdsslim[0] )*(out['sdss'][:,3] < sdsslim[2] )
  out['sdss_ps1'][:,0]=(out['sdss_ps1'][:,0]+0.00128-0.10699*gi+0.00392*gi**2+0.00152*gi**3)*(gi != 0)*(out['sdss'][:,1]!=0)*(out['sdss'][:,1]<sdsslim[0])
  out['sdss_ps1'][:,1]=(out['sdss_ps1'][:,1]-0.00518-0.03561*gi+0.02359*gi**2-0.00447*gi**3)*(gi != 0)*(out['sdss'][:,2]!=0)*(out['sdss'][:,2]<sdsslim[1])
  out['sdss_ps1'][:,2]=(out['sdss_ps1'][:,2]+0.00585-0.01287*gi+0.00707*gi**2-0.00178*gi**3)*(gi != 0)*(out['sdss'][:,3]!=0)*(out['sdss'][:,3]<sdsslim[2])
  out['sdss_ps1'][:,3]=(out['sdss_ps1'][:,3]+0.00144+0.07379*gi-0.03366*gi**2+0.00765*gi**3)*(gi != 0)*(out['sdss'][:,4]!=0)*(out['sdss'][:,4]<sdsslim[3])
  dm = (-out['sdss_ps1'][:,numin]+out['lc_mag'][:,numin,0])
  print np.median(out['sdss_ps1'][:,numin]-out['lc_mag'][:,numin,0])
  dt = ((-out['sdss_mjd']+out['lc_mjd'][:,numin,0])/365.25)
  [total,bins]=np.histogram(dt,bins=12,range=[0,12],weights=dm)
  [total2,bins]=np.histogram(dt,bins=12,range=[0,12],weights=dm**2)
  [num,bins]=np.histogram(dt,bins=12,range=[0,12])
  total=total/num
  std=np.sqrt(total2/num-total**2)
  time=0.5*(bins[1:]+bins[:12])
  return time, total, std, num
  

def test(input=0,plot=1,correct=0,rest=0,L=0,Z=0,constant=0):
  blah=pyfits.open("qso_sdss_ps1_shen.fits")
#  blah=pyfits.open("qso_sdss_ps1.fits")
  out=blah[1].data
  output="output_qso.txt"
  if input == 0:
    if correct > 0:
      ext='_corr'
      gammas=np.array([0.5038556528839815, 0.561977783928414, 0.522906616995358, 0.5310140318599555])
      As=np.array([0.031122285418107583, 0.02101906117378664, 0.01846835815681898, 0.017052938455721686])
    else:
      ext=''
      gammas=np.array([0.479049047593, 0.506137029769, 0.497898596266, 0.440664116035])
      As=np.array([0.0360544022855, 0.0247849407094, 0.0204462230458, 0.0203274924801])
    if constant > 0:
      correct=1
      ext='_constant'
      gammas=np.array([0.5382, 0.5382, 0.5382, 0.5382])
      As=np.array([0.04709, 0.03346, 0.02838, 0.02530])
    if rest > 0:
      correct=1
      ext='_rest'
      gammas=np.array([0.502732892135, 0.550156415017, 0.554237294913, 0.505944551742])
      As=np.array([0.053533732521, 0.0383858564912, 0.0311217236287, 0.0303423741946])
  if input == 1:
    correct=1; rest=1
    zmin=Z*0.5; zmax=zmin+0.5
    lmin=45.0+0.4*L; lmax=lmin+0.4
    ext='_'+str(zmin)+'_'+str(lmin)+'_'+str(zmax)+'_'+str(lmax)+'_rest'
    good= ( (out['REDSHIFT'][:]>zmin) & (out['REDSHIFT'][:]<zmax) & (out['LOGLBOL'][:]>lmin) & (out['LOGLBOL'][:]<lmax)) 
    out=out[good]
    if (L == 0) & (Z == 0 ):
      gammas=np.array([0.384134258829, 0.358139878652, 0.397270714517, 0.473938581124])
      As=np.array([0.0778652473193, 0.0515496966316, 0.036589871641, 0.0335981459357])
    elif (L == 1) & (Z == 0 ):
      gammas=np.array([0.478577027984, 0.514957770884, 0.445583534508, 0.442124102214])
      As=np.array([0.060543761605, 0.0395670478201, 0.0359324788093, 0.0362522727002])
    elif (L == 2) & (Z == 0 ):
      gammas=np.array([0.299502628165, 0.424558307559, 0.418670530177, 0.49462484271])
      As=np.array([0.0675184789056, 0.0440257176673, 0.0353836006979, 0.039997683184])
    elif (L == 1) & (Z == 1 ):
      gammas=np.array([0.488109182183, -0.0301302447883, 0.439008589851, 0.483222172815])
      As=np.array([0.0582883387819, 0.113941947136, 0.0435274932846, 0.0356998080088])
    elif (L == 2) & (Z == 1 ):
      gammas=np.array([0.44103501964, 0.507991555882, 0.587653417712, 0.235369778908])
      As=np.array([0.0472074537688, 0.0396554696823, 0.0336304108823, 0.0451476315785])
    elif (L == 3) & (Z == 1 ):
      gammas=np.array([0.538383600819, 0.370492471669, 0.483492244057, 0.623684184908])
      As=np.array([0.033453641262, 0.034864445464, 0.0300712064712, 0.0215748196614])
    elif (L == 2) & (Z == 2 ):
      gammas=np.array([0.446502470816, 0.523371566307, 0.545894385785, 0.596945857602])
      As=np.array([0.0715048291445, 0.0439020852301, 0.0422657120343, 0.0407693347942])
    elif (L == 3) & (Z == 2 ):
      gammas=np.array([0.504593271346, 0.526833134961, 0.570912159701, 0.655209789073])
      As=np.array([0.0482907205395, 0.0301796564067, 0.0277021049783, 0.0271111958941])
    elif (L == 4) & (Z == 2 ):
      gammas=np.array([0.411414085848, 0.394978885126, 0.3864413504, 0.620476835674])
      As=np.array([0.0384221736053, 0.0272398993163, 0.0245593013434, 0.0199245050025])
    elif (L == 3) & (Z == 3 ):
      gammas=np.array([0.47997524908, 0.567160480205, 0.512759683746, 0.384938627042])
      As=np.array([0.0625898183297, 0.0410679133575, 0.0293472816895, 0.0353512689957])
    elif (L == 4) & (Z == 3 ):
      gammas=np.array([0.496553792616, 0.597178845857, 0.278696345035, 0.572180894252])
      As=np.array([0.042080829893, 0.029702596789, 0.0288585689107, 0.020976481431])
    elif (L == 5) & (Z == 3 ):
      gammas=np.array([0.183280336325, 0.615658001538, 0.195370708934, 0.499908203028])
      As=np.array([0.0416972603472, 0.021754224031, 0.0241371320714, 0.015499604075])
    elif (L == 3) & (Z == 4 ):
      gammas=np.array([0.431275468044, 0.511939871871, 0.362979358826, 0.612025945058])
      As=np.array([0.065237208317, 0.0510733354885, 0.040676220253, 0.0287081050919])
    elif (L == 4) & (Z == 4 ):
      gammas=np.array([0.4990148439, 0.574859679718, 0.528753161228, 0.519218680573])
      As=np.array([0.0451517053403, 0.0360749305303, 0.0271911663346, 0.0217710785802])
    elif (L == 5) & (Z == 4 ):
      gammas=np.array([0.486039429113, 0.681455670798, 0.497923281253, 0.691056759068])
      As=np.array([0.0354794557319, 0.0261187462131, 0.0233881478689, 0.015630449514])
    else:
      return 1;
  good= (( out['sdss_ps1'][:,0]>0) & ( out['mean'][:,0]-out['mean_ap'][:,0]<0.3) & ( out['mean'][:,1]-out['mean_ap'][:,1]<0.3) & ( out['mean'][:,2]-out['mean_ap'][:,2]<0.3) & ( out['mean'][:,3]-out['mean_ap'][:,3]<0.3) & ( out['sdss_type']==6))

  out=out[good]
  out['sdss_ps1'][np.isnan(out['sdss_ps1'])]=0
  out['sdss_ps1_err'][np.isnan(out['sdss_ps1_err'])]=0
  out['sdss_mjd'][np.isnan(out['sdss_mjd'])]=0
  out['lc_mag'][np.isnan(out['lc_mag'])]=0
  out['lc_err'][np.isnan(out['lc_err'])]=0
  out['lc_mjd'][np.isnan(out['lc_mjd'])]=0
  if (rest > 0) | (constant > 0):
    good=(( out['REDSHIFT'][:]>0) & ~np.isnan(out['REDSHIFT'][:]))
    out=out[good]
    out['lc_mjd'][:]= out['lc_mjd'][:]/(1.0+ (out['REDSHIFT'][:]).reshape(np.size(out['REDSHIFT']),1,1))
    out['sdss_mjd'][:]= out['sdss_mjd'][:]/(1.0+ (out['REDSHIFT'][:]))
  plotout=calc_var(out,gammas=gammas,As=As,plot=plot,correct=correct,ext=ext,constant=constant)
  num=4
  np.savetxt(output,np.vstack([out['a'][:,num],out['a_err'][:,num],out['a_chi'][:,num],out['v'][:,num],out['v_err'][:,num],out['v_chi'][:,num],out['tmin'][:,num],out['tmax'][:,num],out['nepoch'][:,num]]).T)
  print '      gammas=np.array(['+str(plotout[0,1])+', '+str(plotout[1,1])+', '+str(plotout[2,1])+', '+str(plotout[3,1])+'])'
  print 'As=np.array(['+str(plotout[0,0])+', '+str(plotout[1,0])+', '+str(plotout[2,0])+', '+str(plotout[3,0])+'])'
  return plotout

def calc_var(out,gammas=np.array([0.479049047593, 0.506137029769, 0.497898596266, 0.440664116035]),As=np.array([0.0360544022855, 0.0247849407094, 0.0204462230458, 0.0203274924801]),correct=0,plot=0,ap=0,kron=0,ext='',constant=0):
  size=out.shape[0]
  if ap:
    aps='_ap'
  elif kron:
    aps='_kron'
  else:
    aps=''
  lcs='lc_mag'+aps
  lces='lc_err'+aps
  lcms='lc_mjd'+aps
  a_s='a'+aps
  aes='a_err'+aps
  acs='a_chi'+aps
  v_s='v'+aps
  ves='v_err'+aps
  vcs='v_chi'+aps
  nes='nepoch'+aps
  mins='tmin'+aps
  maxs='tmax'+aps
  am=np.zeros(2)
  plotout=np.zeros([4,6])
  fscale=(As).reshape(1,4)
  
  var=np.zeros([size,4,30]).astype(np.float64)
  var_wt=np.zeros([size,4,30]).astype(np.float64)
  var_t=np.zeros([size,4,30]).astype(np.float64)
  rrl=np.zeros([size,4,30]).astype(np.float64)
  rrl_wt=np.zeros([size,4,30]).astype(np.float64)

  for num in [0,1,2,3]:
    am[0]=As[num]; am[1]=gammas[num]
    if correct:
      good=((out['sdss_ps1'][:,num] > 0) & ( out['sdss_ps1'][:,num] < 23 ))  
      goodplot=good&(out['mean'][:,num] > 0) & ( out['mean'][:,num] < 23 )
      nuts=((out['sdss_ps1'][:,0]-out['sdss_ps1'][:,2] < 0.2) )
      good2=goodplot.reshape(size,1)
      offset = np.median((out['sdss_ps1'][:,num][goodplot]-out[lcs][:,num,:][good2])[(out[lcs][:,num,:][good2]>0)])
      offset = np.mean((out['sdss_ps1'][:,num][goodplot]-out[lcs][:,num,:][good2])[(out[lcs][:,num,:][good2]>0)])
      print np.sum(good2), offset, np.sum(out[lcs][:,num,:][good2]>0)
      out['sdss_ps1'][:,num]=out['sdss_ps1'][:,num]-offset
    lc_mag=(out[lcs][:,num,:]).astype(np.float64)
    lc_err=(out[lces][:,num,:]).astype(np.float64)
    lc_mjd=(out[lcms][:,num,:]).astype(np.float64)
    sdss_ps1=(out['sdss_ps1'][:,num]).astype(np.float64)
    sdss_ps1_err=(out['sdss_ps1_err'][:,num]).astype(np.float64)
    sdss_mjd=(out['sdss_mjd'][:]).astype(np.float64)
    if ap == 1:
      sdss_ps1=np.ones(np.shape(sdss_ps1))*99.0
      sdss_ps1_err=np.ones(np.shape(sdss_ps1_err))*99.0
      sdss_mjd=np.zeros(np.shape(sdss_mjd))
    lc_nmag=(np.sum(out[lcs][:,num,:]!=0,axis=1))
    fill_in_varinfo(lc_mag,lc_err,lc_mjd,lc_nmag,sdss_ps1,sdss_ps1_err,sdss_mjd,var[:,num],var_wt[:,num,:],var_t[:,num,:],rrl[:,num,:],rrl_wt[:,num,:],am)
    out[aes][:,num]=np.sum(var_wt[:,num,:],axis=1)
    out[ves][:,num]=np.sum(rrl_wt[:,num,:],axis=1)
    out[a_s][:,num]=np.sum(var[:,num,:]*var_wt[:,num,:],axis=1)/(.000000001+out[aes][:,num])
    out[v_s][:,num]=np.sum(rrl[:,num,:]*rrl_wt[:,num,:],axis=1)/(.000000001+out[ves][:,num])
    out[acs][:,num]=np.sum((var[:,num,:]-(out[a_s][:,num]).reshape(size,1))**2*var_wt[:,num,:],axis=1)/(np.sum(var_wt[:,num,:]!=0,axis=1)-.999)*(np.sum(var_wt[:,num,:]!=0,axis=1) > 1)
    out[vcs][:,num]=np.sum((rrl[:,num,:]-(out[v_s][:,num]).reshape(size,1))**2*rrl_wt[:,num,:],axis=1)/(np.sum(rrl_wt[:,num,:]!=0,axis=1)-.999)*(np.sum(rrl_wt[:,num,:]!=0,axis=1) > 1)
    out[nes][:,num]=np.sum(var_t[:,num,:]!=0,axis=1)
    out[mins][:,num]=np.amin(var_t[:,num,:]**(1.0/am[1])+999999999.0*(var_t[:,num,:]==0),axis=1)
    out[mins][:,num][out[mins][:,num]==999999999.0]=0
    out[maxs][:,num]=np.amax(var_t[:,num,:]**(1.0/am[1]),axis=1)
    if plot > 0:
      am[0]=np.sum(var[:,num]*var_wt[:,num])/np.sum(var_wt[:,num])
      plotout[num]= makeplot(var[:,num,:],var_wt[:,num,:],var_t[:,num,:],am,num,ext=ext)
#      print makeplot2(var[:,num,:],var_wt[:,num,:],var_t[:,num,:],am,num,ext=ext)
      
  out[aes][:,4]=np.sum(fscale**2*out[aes][:,:4],axis=1)
  out[ves][:,4]=np.sum(fscale**2*out[ves][:,:4],axis=1)
  out[a_s][:,4]=np.sum(out[a_s][:,:4]*out[aes][:,:4]*fscale.reshape(1,4),axis=1)/(.00000000001+out[aes][:,4])
  out[v_s][:,4]=np.sum(out[v_s][:,:4]*out[ves][:,:4]*fscale.reshape(1,4),axis=1)/(.00000000001+out[ves][:,4])
  out[mins][:,4]=np.amin(out[mins][:,:4]+999999999.0*(out['tmin'][:,:4]==0),axis=1)
  out[mins][:,4][out[mins][:,4]==999999999.0]=0
  out[maxs][:,4]=np.amax(out[maxs][:,:4],axis=1)
  out[nes][:,4]=np.sum(out[nes][:,:4],axis=1)
  out[acs][:,4]=np.sum(((var/fscale.reshape(1,4,1)-(out[a_s][:,4]).reshape(size,1,1))**2*var_wt*fscale.reshape(1,4,1)**2).reshape(size,4*30),axis=1)/(out[nes][:,4]-.999)*(out[nes][:,4]>1)
  out[vcs][:,4]=np.sum(((rrl/fscale.reshape(1,4,1)-(out[v_s][:,4]).reshape(size,1,1))**2*rrl_wt*fscale.reshape(1,4,1)**2).reshape(size,4*30),axis=1)/(out[nes][:,4]-.999)*(out[nes][:,4]>1)
  out[aes][:,:]=np.sqrt(1.0/(.000001+out[aes][:,:])*(out[aes][:,:]>0))
  out[ves][:,:]=np.sqrt(1.0/(.000001+out[ves][:,:])*(out[ves][:,:]>0))
  if constant > 0:
    nf=4
    z=(1+out['REDSHIFT']).reshape([size,1])*np.ones([size,nf])
    L=10.0**((out['LOGLBOL']).reshape([size,1])*np.ones([size,nf])-46.0)
    l= (np.array([.483,.629,.752,.866])[:nf]).reshape([1,nf])/(1+out['REDSHIFT']).reshape([size,1])
    a=(out[a_s][:,:nf]).reshape(size*nf)
    ae=(out[aes][:,:nf]).reshape(size*nf)
    nmag=((out['nmag_ok'][:,:nf]).reshape(size*nf))
    good=(a != 0 ) & (z.reshape(size*nf) < 3.5)  
    z=(z.reshape(size*nf))[good]
    L=(L.reshape(size*nf))[good]
    l=(l.reshape(size*nf))[good]
    nmag=nmag[good]
    ae=ae[good]
    a=a[good]
    np.savetxt('multifit.txt',np.vstack([z,L,l,a,ae,nmag]).T)
  if plot > 0:
    return plotout
#  g1=(out['mean'][:,0]-out['sdss_ps1'][:,0])**2
#  g2=(out['stdev'][:,0]**2-out['err'][:,0]**2)
#  np.savetxt(output, np.vstack([a[:,4],a_err[:,4],a_chi[:,4],v[:,4],v_err[:,4],v_chi[:,4],nepochs[:,4],tmin[:,4],tmax[:,4],g1,g2]).T)
#def fill_in_maginfo(lc_mag,lc_err,lc_mjd,lc_nmag,sdss_ps1,sdss_ps1_err,sdss_mjd,var,var_wt,var_t,rrl,rrl_wt,am, av, av_wt):
def fill_in_varinfo(lc_mag,lc_err,lc_mjd,lc_nmag,sdss_ps1,sdss_ps1_err,sdss_mjd,var,var_wt,var_t,rrl,rrl_wt,am):
	code = \
	"""
	#line 40 "objdata_weave.py"


	// stream through the input arrays
	int size = Nlc_nmag[0];

        double tlim[31]= {	
	.01,0.015848932,0.019952623,0.025118864,0.031622777,
 0.039810717,0.050118723,0.063095734,0.079432823,
	.1,0.15848932,0.19952623,0.25118864,0.31622777,
 0.39810717,0.50118723,0.63095734,0.79432823,
 1.,1.25892541,1.58489319,1.99526231,
 2.51188643,3.16227766,3.98107171,5.01187234,
 6.30957344,7.94328235,10.,12.58925412};

	std::vector<double> dmags, sig, dmjds;
//	std::vector<double> ms;
	for(long i = 0; i < size; i++)
	{
	        double tmax=0;
	        double err2=0; 
	        double dt=0;
		double dm=0;
//		double m = 0;
		double gammam=AM1(1);
		double Am=AM1(0);
// Note that 210 = n(n+1)/2 for n = 20
		double dts[210];
		long ndiffs=0;
		dmags.clear(); sig.clear(); dmjds.clear(); 
//              ms.clear();

	        for(int j = 0; j < LC_NMAG1(i); j++)
		{
		        err2=.0001+SDSS_PS1_ERR1(i)*SDSS_PS1_ERR1(i)+LC_ERR2(i, j)*LC_ERR2(i,j);
		        dt=(LC_MJD2(i, j)-SDSS_MJD1(i))/365.25;
			dm=(LC_MAG2(i, j)-SDSS_PS11(i))*(LC_MAG2(i, j)-SDSS_PS11(i));
//			m=(LC_MAG2(i, j)-SDSS_PS11(i));
			if ( dt < 0 )
			{
			  dt=-1.0*dt;
//			  m=-1.0*m;
			}
			if ( err2 >  0.0001 && err2 < .04 && dt > .01 && dm < 4 && dm > 0 )
			{
			  dts[ndiffs]=dt;
			  ndiffs++;
		          dmags.push_back(dm);
		          sig.push_back(err2 );
		          dmjds.push_back(dt);
//		          ms.push_back(m);
			}  
		}
	        for(int j = 0; j < lc_nmag[i]; j++)
		{
	                for(int k = j+1; k < lc_nmag[i]; k++)
			{
		          err2=LC_ERR2(i,j)*LC_ERR2(i,j)+LC_ERR2(i,k)*LC_ERR2(i,k);
		          dt=(LC_MJD2(i, j)-LC_MJD2(i,k))/365.25;
			  dm=(LC_MAG2(i, j)-LC_MAG2(i,k))*(LC_MAG2(i,j)-LC_MAG2(i,k));
	//		  m=(LC_MAG2(i, j)-SDSS_PS11(i));
			  if ( dt < 0 )
			  {
			    dt=-1.0*dt;
	//		    m=-1.0*m;
			  }
			  if ( err2 > 0.0 && err2 < .04 && dt > .01  && dm < 4 && dm > 0 )
			  {
			    dts[ndiffs]=dt;
			    ndiffs++;
		            dmags.push_back(dm);
		            sig.push_back(err2 );
		            dmjds.push_back(dt);
	//	            ms.push_back(m);
			  }
			}
		}
		tmax=gsl_stats_max (dts, 1, ndiffs);
	        for ( int j=0; j < 30; j++ )
		{
		  double sig0 = 0; double m0 = 0; double msig0 = 0; double sig20=0; double t0 = 0; double mt0=0; double sigt0=0; double npairs=0; double w0=0; double w1 = 0; double mod_wt=0; //double av=0; double av_wt=0;
		  for ( int k=0; k < ndiffs; k++ )
		  {
                    if ( dmjds[k] > tlim[j] && dmjds[k] < tlim[j+1])
	            {
		      dt=pow(dmjds[k],gammam);
		      sig0+=sig[k];
		      m0+=dmags[k];
                      msig0+=dmags[k]*sig[k];
                      sig20+=sig[k]*sig[k];
                      mt0+=dmags[k]*dt;
                      sigt0+=sig[k]*dt;
		      t0+=dt;
		      w0+=0.5/pow(Am+sig[k]/dt,2);
		      w1+=0.5/pow(0.04+sig[k],2);
		      npairs+=1.0;
	//	      av+=ms[k]/(.0001+sig[k]);
	//	      av_wt+=1.0/(.0001+sig[k]);
	            }
		  }
		  if ( npairs > 0.0 )
		  {
		    sig0/=npairs;
		    m0/=npairs;
		    msig0/=npairs;
		    sig20/=npairs;
		    mt0/=npairs;
		    t0/=npairs;
		    w0/=npairs;
	//	    av/=av_wt;
		    dt=pow(t0,1.0/gammam);
		    VAR2(i , j) = (2.0*m0-mt0/t0-sig0)/t0+2.0/t0*(sig0*mt0/t0-msig0)/(Am*t0+sig0);
                    mod_wt =(1.-exp(-dt/tmax*npairs))/(1.-exp(-dt/tmax))/npairs;
                    VAR_WT2(i , j) =mod_wt*w0;
                    VAR_T2(i , j) = t0;
		    RRL2(i , j) = m0-sig0+2.0*(m0*sig0-msig0)/(Am*t0+sig0);
		    RRL_WT2(i , j) = mod_wt*w1;
	//	    AV2(i , j) = av;
	//	    AV_WT2(i , j) = mod_wt*av_wt;

		  }
		  else
		  {
		    VAR2(i , j) = 0;
		    VAR_WT2(i , j) = 0;
		    VAR_T2(i , j) = 0;
		    RRL2(i , j) = 0;
		    RRL_WT2(i , j) = 0;
	//	    AV2(i , j) = 0;
	//	    AV_WT2(i , j) = 0;
		  }
		}

	}
	"""
	inline(code,
                ['lc_mag', 'lc_err', 'lc_mjd', 'lc_nmag', 'sdss_ps1', 'sdss_ps1_err', 'sdss_mjd', 'var', 'var_wt', 'var_t', 'rrl', 'rrl_wt', 'am' ],
#                ['lc_mag', 'lc_err', 'lc_mjd', 'lc_nmag', 'sdss_ps1', 'sdss_ps1_err', 'sdss_mjd', 'var', 'var_wt', 'var_t', 'rrl', 'rrl_wt', 'am', 'av', 'av_wt'],
		headers=['"pmSourceMasks.h"', '<cmath>', '<iostream>', '<vector>', '</a41217d1/morganson/averages2/gsl/gsl_statistics.h>', '<cassert>', '<algorithm>'],
#		libraries=['gsl', 'gslcblas'],
		libraries=[':libgsl.so.0', ':libgslcblas.so.0'],
		library_dirs=['/usr/lib64/'],
		include_dirs=[os.getenv('UBERCAL_DIR')+'/python',os.getenv('UBERCAL_DIR')+'/python/gsl','.','gsl-1.15'],
		undef_macros=['NDEBUG'])
