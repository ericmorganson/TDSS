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
import astropysics,copy
import astropysics.obstools
import calc_var
dustmap='/a77d1/morganson/KP5/DUSTMAP/SFD_dust_4096_%s.fits'
drcoeff=np.array([3.172,2.271,1.682,1.322,1.087]).reshape(1,5)


def e2g(ra,dec):
  if np.size(np.array(ra)) == 0:
    return [[],[]]
  dec0=np.radians(62.871664)
  ra0=np.radians(282.859508)
  dec=np.radians(dec)
  ra=np.radians(ra)
  sinb = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(ra-ra0)*np.sin(dec0)
  sinlm33 = np.sin(dec)*np.sin(dec0)+np.cos(dec)*np.sin(ra-ra0)*np.cos(dec0)
  coslm33 = np.cos(dec)*np.cos(ra-ra0)
  b=np.degrees(np.arcsin(sinb))
  l=np.mod(np.degrees(np.arctan2(sinlm33,coslm33))+32.932,360)
  return [l,b]


def deredden(ra,dec):
  [l,b]=e2g(ra,dec)
  return astropysics.obstools.get_SFD_dust(l,b,dustmap=dustmap)


#import util_efs
testbounds=[ (lsd.bounds.rectangle(30, 0, 31, 1), lsd.bounds.intervalset((-np.inf, np.inf))) ]
def make_average_magnitudes(db, querystr, intostr, zpfun, bounds=None):
	db = lsd.DB(db)
	q = db.query(querystr)
	writer = IntoWriter(db, intostr)
	nrows = 0
	with db.transaction():
		for cell_id, rows in q.execute([(calc_objmag, zpfun, writer)],
					       group_by_static_cell=True,
					       bounds=bounds):
			nrows += len(rows)
	return nrows


joinstr = """
{{
    "m1": [
        "{magtable}",
	"obj_id"
    ],
    "m2": [
        "{magtable}",
	"_id"
    ],
    "type":"indirect"
}}
"""

readnoise = 5.67

def write_join(dbfile, objtable='ps1_obj', magtable=None, clobber=False):
	joinfilename = '.{obj}:{mag}.join'.format(obj=objtable, mag=magtable)
	joinfilename = os.path.join((dbfile.split(':'))[0], joinfilename)
	if os.access(joinfilename, os.F_OK) and not clobber:
		print "File {} exists and clobber not set".format(joinfilename)
		return
	joinfile = open(joinfilename, 'wb')
	joinfile.write(joinstr.format(magtable=magtable))

def ps1_compute_averages(db='/a41217d5/LSD/from_cfa', magtype='psf_inst_mag', bounds=None):
	querystr = ('SELECT _ID, %s as mag, ap_mag as ap_mag, '+
		    'ra, dec, psf_inst_mag_sig as err, '+
		    'filterid, flags, mjd_obs, chip_id, x_psf, y_psf, '+
		    'psf_major, psf_minor, sky, psf_qf, exptime '+
		    'FROM ps1_obj, ps1_det') % magtype
	intostr = 'recalib_mags WHERE obj_id |= obj_id'
	return make_average_magnitudes(db, querystr, intostr,
				       get_zp_ps1, bounds=bounds)


def get_zp_ps1(obj):
	self = get_zp_ps1
	if getattr(self, 'zps', None) is None:
		# Load the zero points
		fn = os.environ['ZPS_ARRAY']
		self.zps = cPickle.load(file(fn))
	zps = self.zps

	# Find the ZP for each MJD. If it does not exist,
	# return +inf
	idx = np.searchsorted(zps.mjd, obj['mjd'])
	idx[idx == len(zps)] = 0
	found = zps.mjd[idx] == obj['mjd']
	print "Found:", sum(found)

	# Join the ZPs
	zp = np.zeros(len(idx), dtype='f4')
	zp[:] = zps.ZP[idx]

	# Remove those that are flagged bad
	bad = ~found
	bad |= zps.time_to_bad[idx] < 30. / (24*60)
	zp[bad] = np.nan

	# Apply flat field corrections
	zp += cal.flat_offs(obj.chip_id, obj.x_psf, obj.y_psf,
			    obj.mjd_obs, obj.filterid)

	print "OK/Not OK:", sum(~bad), sum(bad)

	return zp

def fill_in_maginfo(id_out, ndet, ndet_ok, nmag, nmag_ok, mean, stdev,
		    merr, median, q25, q75, lc_mag, lc_err, lc_mjd, lc_psf, lc_dra, lc_ddec, c_chi, id_in, mags, errs, band, flags,
		    zp, psf_qf, mjd_obs, psfs, dras, ddecs):
	code = \
	"""
	#line 93 "objdata_weave.py"

	assert(Sid_out[0] == sizeof(*id_out));	// Make sure we've got a contiguous array

	uint32_t bad = PM_SOURCE_MODE_FAIL | PM_SOURCE_MODE_POOR | PM_SOURCE_MODE_SATSTAR | 
		PM_SOURCE_MODE_BLEND | PM_SOURCE_MODE_EXTERNAL | PM_SOURCE_MODE_BADPSF |
		PM_SOURCE_MODE_DEFECT | PM_SOURCE_MODE_SATURATED | PM_SOURCE_MODE_CR_LIMIT 
		// | 
		// PM_SOURCE_MODE_EXT_LIMIT | PM_SOURCE_MODE_MOMENTS_FAILURE | 
		// PM_SOURCE_MODE_SIZE_SKIPPED | PM_SOURCE_MODE_BIG_RADIUS
		;

	// stream through the input arrays
	int size = Nmags[0];
	std::vector<double> mags1, wt, mjds, mags2, psfs1, dras1, ddecs1;
	for(int i = 0, j = 0; i != size; i = j)
	{
		j = i;
		mags1.clear(); wt.clear(); mjds.clear(); mags2.clear(); psfs1.clear(); dras1.clear(); ddecs1.clear();
		while(j != size && id_in[i] == id_in[j] && band[i] == band[j])
		{
			float mag = MAGS1(j);
			float err = ERRS1(j);
			float psf_qf = PSF_QF1(j);
			float zp = ZP1(j);
			double mjd = MJD_OBS1(j);
			float psf = PSFS1(j);
			float dra = DRAS1(j);
			float ddec = DDECS1(j);
			uint32_t flag = FLAGS1(j);
			//if(std::isfinite(mag) && (flag & PM_SOURCE_MODE_FITTED))

			if(std::isfinite(mag) && std::isfinite(zp) && ((flag & bad) == 0) && (psf_qf > 0.85))
			{
				mag += zp;
				mags1.push_back(mag);
				mags2.push_back(mag*mag);
				mjds.push_back(mjd);
				psfs1.push_back(psf);
				dras1.push_back(dra);
				ddecs1.push_back(ddec);
				wt.push_back(1. / (err*err + 0.01*0.01)); // add 0.01 in quadrature to compensate for unrealistically small quoted errors
			}
			j++;
		}

		// find where to store
		int row = std::lower_bound(id_out, id_out + Nid_out[0], id_in[i]) - id_out;
		int col = band[i];
		assert(id_out[row] == id_in[i]);
		assert(0 <= row && row < Nid_out[0]);
		assert(0 <= col && col < 5);

		// store number of elements (all and finite)
		   NMAG2(row, col) = j - i;
		NMAG_OK2(row, col) = mags1.size();

		if(!mags1.empty())
		{
			// median statistics
			Q252(row, col)    = gsl_stats_quantile_from_sorted_data(&mags1[0], 1, mags1.size(), 0.25);
			MEDIAN2(row, col) = gsl_stats_quantile_from_sorted_data(&mags1[0], 1, mags1.size(), 0.50);
			Q752(row, col)    = gsl_stats_quantile_from_sorted_data(&mags1[0], 1, mags1.size(), 0.75);

			// mean statistics
			MEAN2(row, col)  = gsl_stats_wmean(&wt[0], 1, &mags1[0], 1, mags1.size());
			STDEV2(row, col) = fabs(gsl_stats_wsd(&wt[0], 1, &mags1[0], 1, mags1.size()));	// I wrap it in fabs because for N=0 it returns a -0 (??)

			// mean error computed as 1./sum(wts)
			double w = 0.;
			for(int i = 0; i != wt.size(); i++) { w += wt[i]; }
			MERR2(row, col) = 1. / sqrt(w);
			// Computing Chi^2 of a constant source assumption
			C_CHI2(row, col) = w*(gsl_stats_wmean(&wt[0], 1, &mags2[0], 1, mags2.size()) -MEAN2(row, col)*MEAN2(row, col));
		}
                // store the light curves
                for(unsigned k = 0; k != std::min(mags1.size(), (size_t)20); k++)
                {
                      LC_MAG3(row, col, k) = mags1[k];
                      LC_ERR3(row, col, k) = sqrt(1. / wt[k]);
                      LC_MJD3(row, col, k) = mjds[k];
                      LC_PSF3(row, col, k) = psfs1[k];
                      LC_DRA3(row, col, k) = dras1[k];
                      LC_DDEC3(row, col, k) = ddecs1[k];
                 }

	}
	"""
	inline(code,
		['id_out', 'ndet', 'ndet_ok', 'nmag', 'nmag_ok', 'mean', 'stdev', 'merr', 'median', 'q25', 'q75', 'lc_mag', 'lc_err', 'lc_mjd', 'lc_psf', 'lc_dra', 'lc_ddec', 'c_chi', 'id_in', 'mags', 'errs', 'band', 'flags', 'zp', 'psf_qf', 'mjd_obs', 'psfs', 'dras', 'ddecs'],
		headers=['"pmSourceMasks.h"', '<cmath>', '<iostream>', '<vector>', '<gsl/gsl_statistics.h>', '<cassert>', '<algorithm>'],
#		libraries=['gsl', 'gslcblas'],
		libraries=[':libgsl.so.0', ':libgslcblas.so.0'],
		library_dirs=['/usr/lib64/'],
		include_dirs=[os.getenv('UBERCAL_DIR')+'/python',os.getenv('UBERCAL_DIR')+'/python/gsl','.','gsl-1.15'],
		undef_macros=['NDEBUG'])

def calc_objmag_aux(all_rows, qresult, zpfun, qwriter):
	all_rows.sort(["filterid", "_ID", "mag"])
	zp = zpfun(all_rows)
	zp = np.array(zp, dtype='float32')
	unneeded_columns = ['chip_id', 'x_psf', 'y_psf', 'airmass', 'smf_fn']
	needed_columns = ([x for x in all_rows.keys()
			   if x not in unneeded_columns])
	all_rows = all_rows[needed_columns] # try to free some memory

	# Prepare the output array
	objs, idx = np.unique(all_rows['_ID'], return_index=True)
	out = colgroup.ColGroup(
		dtype=[
			('obj_id', 'u8'),
			('ra', 'f8'), ('dec', 'f8'),
			('ndet', 'i2'), ('ndet_ok', 'i2'),
			('nmag', '5i2'), ('nmag_ok', '5i2'),
			('mean', '5f4'), ('stdev', '5f4'), ('err', '5f4'),
			('median', '5f4'), ('q25', '5f4'), ('q75', '5f4'),
			('ndet_ap', 'i2'), ('ndet_ap_ok', 'i2'),
			('nmag_ap', '5i2'), ('nmag_ap_ok', '5i2'),
			('mean_ap', '5f4'), ('stdev_ap', '5f4'),
			('err_ap', '5f4'),
			('median_ap', '5f4'), ('q25_ap', '5f4'), ('q75_ap', '5f4'),
			('ndet_kron', 'i2'), ('ndet_kron_ok', 'i2'),
			('nmag_kron', '5i2'), ('nmag_kron_ok', '5i2'),
			('mean_kron', '5f4'), ('stdev_kron', '5f4'),
			('err_kron', '5f4'),
			('median_kron', '5f4'), ('q25_kron', '5f4'), ('q75_kron', '5f4'),
                        ('lc_mag', '(5,20)f4'), ('lc_err', '(5,20)f4'), ('lc_mjd', '(5,20)f8'),
                        ('lc_psf', '(5,20)f4'), ('lc_dra', '(5,20)f4'), ('lc_ddec', '(5,20)f8'),
			('lc_mag_ap', '(5,20)f4'), ('lc_err_ap', '(5,20)f4'), ('lc_mjd_ap', '(5,20)f8'),
                        ('lc_psf_ap', '(5,20)f4'), ('lc_dra_ap', '(5,20)f4'), ('lc_ddec_ap', '(5,20)f8'),
			('lc_mag_kron', '(5,20)f4'), ('lc_err_kron', '(5,20)f4'), ('lc_mjd_kron', '(5,20)f8'),
                        ('lc_psf_kron', '(5,20)f4'), ('lc_dra_kron', '(5,20)f4'), ('lc_ddec_kron', '(5,20)f8'),
			('dr', '5f4'), ('ebv', 'f4'), ('sdss', '5f4'), ('sdss_err', '5f4'), 
			('sdss_dr', '5f4'), ('sdss_mjd', 'f8'), ('sdss_type', 'i2'), ('sdss_ps1','5f4'),
			('sdss_ps1_err','5f4'), 
			('c_chi','5f4'), ('c_chi_t','f4'), 
			('c_chi_ap','5f4'), ('c_chi_t_ap','f4'), 
			('c_chi_kron','5f4'), ('c_chi_t_kron','f4'), 
			('a','5f4'), ('a_err','5f4'), ('a_chi','5f4'),
			('v','5f4'), ('v_err','5f4'), ('v_chi','5f4'),
			('tmin','5f4'), ('tmax','5f4'), ('nepoch','5i4'),
			('a_ap','5f4'), ('a_err_ap','5f4'), ('a_chi_ap','5f4'),
			('v_ap','5f4'), ('v_err_ap','5f4'), ('v_chi_ap','5f4'),
			('tmin_ap','5f4'), ('tmax_ap','5f4'), ('nepoch_ap','5i4'),
			('a_kron','5f4'), ('a_err_kron','5f4'), ('a_chi_kron','5f4'),
			('v_kron','5f4'), ('v_err_kron','5f4'), ('v_chi_kron','5f4'),
			('tmin_kron','5f4'), ('tmax_kron','5f4'), ('nepoch_kron','5i4')
#			('median_ap', '5f4'), ('q25_ap', '5f4'), ('q75_ap', '5f4'),
#			('maglimit', '5f4')
			],
		size=len(objs)
		)
	out['obj_id'][:] = objs
	out['ra'][:]  = all_rows['ra'][idx]
	out['dec'][:] = all_rows['dec'][idx]
        out['sdss'][:] = np.vstack([all_rows['u'][idx],all_rows['g'][idx],all_rows['r'][idx],all_rows['i'][idx],all_rows['z'][idx]]).T        
        out['sdss_err'][:] = np.vstack([all_rows['uErr'][idx],all_rows['gErr'][idx],all_rows['rErr'][idx],all_rows['iErr'][idx],all_rows['zErr'][idx]]).T        
        out['sdss_dr'][:] = np.vstack([all_rows['uExt'][idx],all_rows['gExt'][idx],all_rows['rExt'][idx],all_rows['iExt'][idx],all_rows['zExt'][idx]]).T        
	out['sdss_mjd'][:] = all_rows['mjd_s'][idx]
	out['sdss_type'][:] = all_rows['type'][idx]
	numout=(out['ra'][:]).size
	blank=np.zeros(numout)
        out['sdss_ps1'][:] = np.vstack([all_rows['g_sp'][idx],all_rows['r_sp'][idx],all_rows['i_sp'][idx],all_rows['z_sp'][idx], blank]).T        
        out['sdss_ps1_err'][:] = np.vstack([all_rows['gErr'][idx],all_rows['rErr'][idx],all_rows['iErr'][idx],all_rows['zErr'][idx], blank]).T        
	# Pull out the arrays we'll be using
	(id_out, ra, dec, ndet, ndet_ok, nmag, nmag_ok,
	 mean, stdev, merr, median, q25, q75,
	 ndet_ap, ndet_ap_ok, nmag_ap, nmag_ap_ok,
	 mean_ap, stdev_ap, merr_ap, median_ap, q25_ap, q75_ap,
#	 mean_ap, stdev_ap, merr_ap, median_ap, q25_ap, q75_ap,
#	 maglimit) = out.as_columns()
	 ndet_kron, ndet_kron_ok, nmag_kron, nmag_kron_ok,
	 mean_kron, stdev_kron, merr_kron, median_kron, q25_kron, q75_kron,
	 lc_mag, lc_err, lc_mjd, lc_psf, lc_dra, lc_ddec, 
	 lc_mag_ap, lc_err_ap, lc_mjd_ap, lc_psf_ap, lc_dra_ap, lc_ddec_ap,
	 lc_mag_kron, lc_err_kron, lc_mjd_kron, lc_psf_kron, lc_dra_kron, lc_ddec_kron,
	 dr,ebv,sdss,sdss_err,sdss_dr,sdss_mjd,sdss_type,sdss_ps1,sdss_ps1_err,
         c_chi, c_chi_t, 
	 c_chi_ap, c_chi_t_ap,
	 c_chi_kron, c_chi_t_kron,
	 a, a_err, a_chi, v, v_err, v_chi, tmin, tmax, nepoch,
	 a_ap, a_err_ap, a_chi_ap, v_ap, v_err_ap, v_chi_ap, tmin_ap, tmax_ap, nepoch_ap,
	 a_kron, a_err_kron, a_chi_kron, v_kron, v_err_kron, v_chi_kron, tmin_kron, tmax_kron, nepoch_kron) = out.as_columns()
	id_in, mags, errs, filterid, flags, psf_qf, ap_mags, kron_mags, kron_errs, mjd_obs, psfs, dras, ddecs = all_rows['_ID'], all_rows['mag'], all_rows['err'], all_rows['filterid'], all_rows['flags'], all_rows['psf_qf'], all_rows['ap_mag'], all_rows['kron_mag'], all_rows['kron_err'], all_rows['mjd_obs'], np.sqrt(0.5*(all_rows['psf_major']**2+all_rows['psf_minor']**2)), np.cos(np.radians(all_rows['dec']))*(all_rows['ras']-all_rows['ra'])*3600., (all_rows['decs']-all_rows['dec'])*3600.

	# Join the zero-point information
#	immaglimit = np.zeros(len(all_rows))
	check_pos = [all_rows['sky'], all_rows['psf_major'],
		     all_rows['psf_minor'], all_rows['exptime']]
	check_finite = check_pos + [zp]
	mask = np.ones(len(all_rows), dtype='bool')
	for col in check_finite:
		mask = mask & np.isfinite(col)
	olderr = np.seterr(invalid='ignore')
	for col in check_pos:
		mask = mask & (col > 0)
	np.seterr(**olderr)
#	immaglimit[mask] = (4.*np.pi*(all_rows['sky'][mask]+readnoise**2.)*
#			    all_rows['psf_major'][mask]*
#			    all_rows['psf_minor'][mask]/
#			    (8*np.log(2)))
#	immaglimit[mask] = np.sqrt(immaglimit[mask])/all_rows['exptime'][mask]
#	mask = mask & (immaglimit > 0)
#	immaglimit[mask] = -2.5*np.log10(5.*immaglimit[mask])+zp[mask]
#	immaglimit[~mask] = np.nan
#	keys = util_efs.unique_multikey(all_rows, ['filterid', '_ID'])
#	maximmaglimit = util_efs.max_bygroup(immaglimit, keys)
#	npkey = np.zeros(len(keys), dtype='i4')
#	npkey[0] = keys[0]+1
#	npkey[1:] = keys[1:]-keys[0:-1]
#	okeys = np.repeat(np.arange(len(keys), dtype='i4'), npkey)
#	immaglimit = maximmaglimit[okeys]
	# 5 sigma magnitude limit if: Gaussian PSF, described by psf_major and psf_minor; background limited
	# (read noise = 5.7 ; no shot noise)

	# Convert filterid to index
	band = np.empty(len(all_rows), dtype='i4')
	for f, i in { 'g.0000': 0, 'r.0000': 1, 'i.0000': 2, 'z.0000': 3, 'y.0000': 4 }.iteritems():
		band[filterid == f] = i

	fill_in_maginfo(id_out, ndet, ndet_ok, nmag, nmag_ok, mean, stdev,
			merr, median, q25, q75, lc_mag, lc_err, lc_mjd, 
			lc_psf, lc_dra, lc_ddec, c_chi, id_in, mags, errs, band, flags,
			zp, psf_qf, mjd_obs, psfs, dras, ddecs )
	fill_in_maginfo(id_out, ndet_ap, ndet_ap_ok, nmag_ap, nmag_ap_ok,
			mean_ap, stdev_ap, merr_ap, median_ap, q25_ap, q75_ap, lc_mag_ap, lc_err_ap, lc_mjd_ap,
                        lc_psf_ap, lc_dra_ap, lc_ddec_ap,
			c_chi_ap, id_in, ap_mags, errs, band, flags, zp, psf_qf, mjd_obs, psfs, dras, ddecs)
	fill_in_maginfo(id_out, ndet_kron, ndet_kron_ok, nmag_kron, nmag_kron_ok,
			mean_kron, stdev_kron, merr_kron, median_kron, q25_kron, q75_kron, lc_mag_kron, lc_err_kron, lc_mjd_kron,
			lc_psf_kron, lc_dra_kron, lc_ddec_kron,
			c_chi_kron, id_in, kron_mags, kron_errs, band, flags, zp, psf_qf, mjd_obs, psfs, dras, ddecs)
	# Compute ndet
	out['ndet'][:] = np.sum(out['nmag'], axis=1)
	assert np.all(out['ndet'])
	out['ndet_ok'][:] = np.sum(out['nmag_ok'], axis=1)
        out['c_chi_t'][:] = np.sum(out['c_chi'], axis=1)
        out['c_chi'][:,:]=out['c_chi'][:,:]/(out['nmag_ok'][:,:]-0.999999)*(out['nmag_ok'][:,:]>1)
        ndof=out['ndet_ok'][:]-np.sum((out['nmag_ok']>0), axis=1)
	out['c_chi_t'][:] = out['c_chi_t'][:]/(ndof+0.000001)*(ndof>0) 

	out['ndet_ap'][:] = np.sum(out['nmag_ap'], axis=1)
	out['ndet_ap_ok'][:] = np.sum(out['nmag_ap_ok'], axis=1)
        out['c_chi_t_ap'][:] = np.sum(out['c_chi_ap'], axis=1)
        out['c_chi_ap'][:,:]=out['c_chi_ap'][:,:]/(out['nmag_ap_ok'][:,:]-0.999999)*(out['nmag_ap_ok'][:,:]>1)
        ndof=out['ndet_ap_ok'][:]-np.sum((out['nmag_ap_ok']>0), axis=1)
	out['c_chi_t_ap'][:] = out['c_chi_t_ap'][:]/(ndof+0.000001)*(ndof>0) 

	out['ndet_kron'][:] = np.sum(out['nmag_kron'], axis=1)
	out['ndet_kron_ok'][:] = np.sum(out['nmag_kron_ok'], axis=1)
        out['c_chi_t_kron'][:] = np.sum(out['c_chi_kron'], axis=1)
        out['c_chi_kron'][:,:]=out['c_chi_kron'][:,:]/(out['nmag_kron_ok'][:,:]-0.999999)*(out['nmag_kron_ok'][:,:]>1)
        ndof=out['ndet_kron_ok'][:]-np.sum((out['nmag_kron_ok']>0), axis=1)
	out['c_chi_t_kron'][:] = out['c_chi_t_kron'][:]/(ndof+0.000001)*(ndof>0) 

        # Compute variability statistics
	calc_var.calc_var(out)
	calc_var.calc_var(out,ap=1)
	calc_var.calc_var(out,kron=1)

        # Compute ebv
	out['ebv'][:] = deredden(out['ra'][:],out['dec'][:])
	out['dr'][:] = np.dot((out['ebv'][:]).reshape((out['ebv'][:]).size ,1),drcoeff )

	# Write out the result
	try:
		result = qwriter.write(qresult.static_cell, out)
	except Exception as e:
		print len(out)
                print out.dtype
		print out[0]
		raise e
	return result


def calc_objmag(qresult, zpfun, qwriter):
	"""
	Compute object magnitude from detections.

	Compute ndet, median, average, SIQR per detection.
	"""
	# PS1 bug workaround: Ignore the few objects that wound up on the south pole
	res = (0, [])
	if qresult.static_cell & 0xFFFFFFFF00000000 == 0:
		logging.warning("Encountered the cell at the south pole. Dropping it.")
		yield res

	if True:
		for all_rows in colgroup.partitioned_fromiter(qresult, "_ID", 5*1000*1000, blocks=True):
			if len(all_rows) == 0:
				continue
			res = calc_objmag_aux(all_rows, qresult, zpfun, qwriter)
			all_rows = None
	else:
		all_rows = colgroup.fromiter(qresult, blocks=True)
		if len(all_rows) == 0:
			return
		res = calc_objmag_aux(all_rows, qresult, zpfun, qwriter)
		all_rows = None
	yield res
