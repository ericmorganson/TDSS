import ubercal_flat
import os
import cPickle
from make_average_magnitudes import make_average_magnitudes
from make_average_magnitudes import write_join
import pdb
import numpy
import lsd
from lsd import colgroup
import pdb
from lsd import bounds
testbounds=[ (lsd.bounds.beam(35.875,-4.250,1.5), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
testbounds=[ (lsd.bounds.beam(130.592,44.317,1.5), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
testbounds=[ (lsd.bounds.beam(150.000,02.200,1.5), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
testbounds=[ (lsd.bounds.beam(334.118,00.283,1.5), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
testbounds=[ (lsd.bounds.rectangle(329,5,330,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
testbounds=[ (lsd.bounds.rectangle(328,-6,330,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(338,-6,340,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(330,-6,338,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(329.8,5.0,329.9,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(329.6,5.0,329.8,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(328,0,340,6), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(328,-6,340,0), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.beam(180.0,30.0,0.1), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, 10, 360, -10, coordsys='gal'), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, 0, 45, 90), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(45, 0, 360, 90), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, -30, 90, 0), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(90, -30, 180, 0), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(180, -30, 360, 0), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, -30, 360, 90), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(308, -1.3, 60, 1.3), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, -30, 270, 90), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
#testbounds=[ (lsd.bounds.rectangle(0, -5, 360, 30), lsd.bounds.intervalset((-numpy.inf, numpy.inf))) ]
input='ps1'
def get_zp_ucal(obj, sol=None):
    self = get_zp_ucal
    if getattr(self, 'sol', None) is None:
        raise Exception('Must set sol first.')
    sol = self.sol
    filterid = obj['filterid'][0][0]
    zp = ubercal_flat.apply_flat_solution(obj, sol)
    if sum(numpy.isfinite(zp)) == 0:
        # pdb.set_trace()
        pass
    return zp

def make_zpfun_ucal(dir):
    sol = ubercal_flat.read_flat_solution(dir)
    zpfun = get_zp_ucal
    zpfun.sol = sol
    return zpfun

def compute_averages_ucal(db=None, dir='ucalqw_noref.fits', magtype='psf_inst_mag',
                          bounds=testbounds, magtablename=input+'_sdss_ps1',objtable=input+'_obj_sdss'):
    if dir is None:
        raise Exception('Must set ucal dir')
    if db is None:
        db = os.environ['LSD_DB']
#    querystr = ('SELECT _ID, %s as mag, ap_mag, ra, '+
    querystr = ('SELECT _ID, obj_id, %s as mag, ap_mag,'+ 
                '-2.5*log10(kron_flux+numpy.spacing(1)) +mag-psf_inst_mag as kron_mag, kron_flux_err/(kron_flux+numpy.spacing(1)) as kron_err, ra, '+
                'dec, psf_inst_mag_sig as err, '+
                'type, '+
                'u, uErr, uExt, '+
                'g, gErr, gExt, '+
                'r, rErr, rExt, '+
                'i, iErr, iExt, '+
                'z, zErr, zExt, '+
                'g_sp, r_sp, i_sp, z_sp, mjd_s, '+
                'filterid, flags, mjd_obs, chip_id, x_psf, y_psf, airmass, '+
                'smf_fn, psf_qf, exptime, sky, psf_major, psf_minor, ps1_det.ra as ras, ps1_det.dec as decs '+
                'FROM '+objtable+', ps1_det(matchedto='+objtable+',nmax=20), ps1_exp WHERE (exptime < 60)') % magtype
#                'FROM '+objtable+', ps1_det, ps1_exp WHERE (exptime < 60)') % magtype
    intostr = ('%s(spatial_keys=[ra, dec]) WHERE obj_id |= obj_id'
               % magtablename)
    zpfun = make_zpfun_ucal(dir)
    olderr = numpy.seterr(under='ignore')
    nrow =  make_average_magnitudes(db, querystr, intostr,
                                    zpfun, bounds=bounds)
    print nrow
    numpy.seterr(**olderr)
    write_join(db,objtable=objtable , magtable=magtablename, clobber=True)
