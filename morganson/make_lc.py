#!/usr/bin/env python

import pyfits, sys
import numpy as np, matplotlib,os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def makeplot(inname="qso_sdss_ps1_shen.fits",n=1):
  blah=pyfits.open(inname)[1].data
  g=blah['lc_mag'][n,0,:]; g_err=blah['lc_err'][n,0,:]; g_mjd=blah['lc_mjd'][n,0,:]
  gs=blah['sdss_ps1'][n,0]; gs_err=blah['sdss_ps1_err'][n,0]; gs_mjd=blah['sdss_mjd'][n]
  r=blah['lc_mag'][n,1,:]; r_err=blah['lc_err'][n,1,:]; r_mjd=blah['lc_mjd'][n,1,:]
  rs=blah['sdss_ps1'][n,1]; rs_err=blah['sdss_ps1_err'][n,1]; rs_mjd=blah['sdss_mjd'][n]
  ip=blah['lc_mag'][n,2,:]; i_err=blah['lc_err'][n,2,:]; i_mjd=blah['lc_mjd'][n,2,:]
  isp=blah['sdss_ps1'][n,2]; is_err=blah['sdss_ps1_err'][n,2]; is_mjd=blah['sdss_mjd'][n]
  z=blah['lc_mag'][n,3,:]; z_err=blah['lc_err'][n,3,:]; z_mjd=blah['lc_mjd'][n,3,:]
  zs=blah['sdss_ps1'][n,3]; zs_err=blah['sdss_ps1_err'][n,3]; zs_mjd=blah['sdss_mjd'][n]
  mags=np.hstack([g,gs,r,rs,ip,isp,z,zs])
  mjds=np.hstack([g_mjd,gs_mjd,r_mjd,rs_mjd,i_mjd,is_mjd,z_mjd,zs_mjd])
  mjds=mjds[(mags > 0) & (mags < 25)]
  mags=mags[(mags > 0) & (mags < 25)]
  minmags = np.round(np.min(mags)-0.15,1)
  maxmags = np.round(np.max(mags)+0.15,1)
  minmjds = np.round(np.min(mjds)-50,-2)
  maxmjds = np.round(np.max(mjds)+50,-2)
  outname=os.path.splitext(inname)[0]+"_"+str(n)+".png"
  print outname
  plt.figure(figsize=(8,7.2))
  plt.subplot(111)
  plt.rcParams['font.size'] = 20
  plt.xlim((minmjds,maxmjds))
  if maxmjds-minmjds > 2500:
     xticks= np.arange(50000,60000,1000)
  else:
     xticks= np.arange(50000,60000,500)
  xticks=xticks[(xticks>minmjds) & (xticks<maxmjds)]
  plt.xticks(xticks,np.array(xticks).astype('S10'))
  #plt.xticks([55400,55600,55800,56000,56200],['55400','55600','55800','56000','56200'])
  plt.ylim((minmags,maxmags))
  #plt.yticks([19.4,19.6,19.8,20.0,20.2],['19.4','19.6','19.8','20.0','20.2'])
  plt.errorbar(g_mjd,g,yerr=g_err,fmt='og')  
  plt.errorbar(r_mjd,r,yerr=r_err,fmt='or')  
  plt.errorbar(i_mjd,ip,yerr=i_err,fmt='oc')  
  plt.errorbar(z_mjd,z,yerr=z_err,fmt='ob')  
  plt.errorbar(gs_mjd,gs,yerr=gs_err,fmt='og')  
  plt.errorbar(rs_mjd,rs,yerr=rs_err,fmt='or')  
  plt.errorbar(is_mjd,isp,yerr=is_err,fmt='oc')  
  plt.errorbar(zs_mjd,zs,yerr=zs_err,fmt='ob')  
  plt.xlabel('MJD')
  plt.ylabel('Mag')
  xs=np.arange(minmjds,maxmjds)
  gys=np.ones(xs.size)*gs
  rys=np.ones(xs.size)*rs
  iys=np.ones(xs.size)*isp
  zys=np.ones(xs.size)*zs
  plt.plot(xs,gys,'-g')  
  plt.plot(xs,rys,'-r')  
  plt.plot(xs,iys,'-c')  
  plt.plot(xs,zys,'-b')  
  plt.savefig(outname)
  plt.clf()

for n in np.array(sys.argv[2:]).astype(int):
  makeplot(inname=sys.argv[1],n=n)
