




// DO NOT EDIT !
// This file is generated using the MantaFlow preprocessor (prep generate).




#line 1 "/home/ubuntu/tempoGAN/source/plugin/tfplugins.cpp"
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2017 Kiwon Um, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * tensorflor/numpy plugins, mostly for MLFLIP for now  (only compiled if NUMPY is enabled)
 *
 ******************************************************************************/

#include "levelset.h"
#include "commonkernels.h"
#include "particle.h"
#include <cmath>

using namespace std;

namespace Manta {




//! simple test kernel and kernel with numpy array 


 struct knSimpleNumpyTest : public KernelBase { knSimpleNumpyTest(Grid<Real>& grid, PyArrayContainer npAr, Real scale) :  KernelBase(&grid,0) ,grid(grid),npAr(npAr),scale(scale)   { runMessage(); run(); }  inline void op(int i, int j, int k, Grid<Real>& grid, PyArrayContainer npAr, Real scale )  {
	const float* p = reinterpret_cast<float*>(npAr.pData);
	grid(i,j,k) += scale * (Real)p[j*grid.getSizeX()+i]; // calc access into numpy array, no size check here!
}   inline Grid<Real>& getArg0() { return grid; } typedef Grid<Real> type0;inline PyArrayContainer& getArg1() { return npAr; } typedef PyArrayContainer type1;inline Real& getArg2() { return scale; } typedef Real type2; void runMessage() { debMsg("Executing kernel knSimpleNumpyTest ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {  const int _maxX = maxX; const int _maxY = maxY; if (maxZ > 1) { 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int k=minZ; k < maxZ; k++) for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,grid,npAr,scale);  } } else { const int k=0; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (int j=0; j < _maxY; j++) for (int i=0; i < _maxX; i++) op(i,j,k,grid,npAr,scale);  } }  } Grid<Real>& grid; PyArrayContainer npAr; Real scale;   };
#line 29 "plugin/tfplugins.cpp"



//! simple test function and kernel with numpy array 
void simpleNumpyTest( Grid<Real>& grid, PyArrayContainer npAr, Real scale) {
	knSimpleNumpyTest(grid, npAr, scale);
} static PyObject* _W_0 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "simpleNumpyTest" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; Grid<Real>& grid = *_args.getPtr<Grid<Real> >("grid",0,&_lock); PyArrayContainer npAr = _args.get<PyArrayContainer >("npAr",1,&_lock); Real scale = _args.get<Real >("scale",2,&_lock);   _retval = getPyNone(); simpleNumpyTest(grid,npAr,scale);  _args.check(); } pbFinalizePlugin(parent,"simpleNumpyTest", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("simpleNumpyTest",e.what()); return 0; } } static const Pb::Register _RP_simpleNumpyTest ("","simpleNumpyTest",_W_0);  extern "C" { void PbRegister_simpleNumpyTest() { KEEP_UNUSED(_RP_simpleNumpyTest); } } 



//! extract feature vectors




 struct knExtractFeatureVel : public KernelBase { knExtractFeatureVel( const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const MACGrid &vel, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window) :  KernelBase(p.size()) ,p(p),fv(fv),N_row(N_row),off_begin(off_begin),vel(vel),scale(scale),ptype(ptype),exclude(exclude),window(window)   { runMessage(); run(); }   inline void op(IndexInt idx,  const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const MACGrid &vel, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window )  {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (vel.is3D()) ? -window : 0, K = (vel.is3D()) ? window : 0;
	const IndexInt D = (vel.is3D()) ? 3 : 2;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k));
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Vec3 vel_s = vel.getInterpolated(pos_s)*scale;
				const IndexInt off_vel = off_idx + off_begin + off_stencil*D;
				fv[off_vel + 0] = vel_s[0];
				fv[off_vel + 1] = vel_s[1];
				if(vel.is3D()) fv[off_vel + 2] = vel_s[2];

				++off_stencil;
			}
		}
	}
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline Real* getArg1() { return fv; } typedef Real type1;inline const IndexInt& getArg2() { return N_row; } typedef IndexInt type2;inline const IndexInt& getArg3() { return off_begin; } typedef IndexInt type3;inline const MACGrid& getArg4() { return vel; } typedef MACGrid type4;inline const Real& getArg5() { return scale; } typedef Real type5;inline const ParticleDataImpl<int> * getArg6() { return ptype; } typedef ParticleDataImpl<int>  type6;inline const int& getArg7() { return exclude; } typedef int type7;inline const int& getArg8() { return window; } typedef int type8; void runMessage() { debMsg("Executing kernel knExtractFeatureVel ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,fv,N_row,off_begin,vel,scale,ptype,exclude,window);  }   } const BasicParticleSystem& p; Real* fv; const IndexInt N_row; const IndexInt off_begin; const MACGrid& vel; const Real scale; const ParticleDataImpl<int> * ptype; const int exclude; const int window;   };
#line 46 "plugin/tfplugins.cpp"






 struct knExtractFeaturePhi : public KernelBase { knExtractFeaturePhi( const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const Grid<Real> &phi, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window) :  KernelBase(p.size()) ,p(p),fv(fv),N_row(N_row),off_begin(off_begin),phi(phi),scale(scale),ptype(ptype),exclude(exclude),window(window)   { runMessage(); run(); }   inline void op(IndexInt idx,  const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const Grid<Real> &phi, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window )  {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (phi.is3D()) ? -window : 0, K = (phi.is3D()) ? window : 0;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k));
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Real phi_s = phi.getInterpolated(pos_s)*scale;
				const IndexInt off_phi = off_idx + off_begin + off_stencil;
				fv[off_phi] = phi_s;

				++off_stencil;
			}
		}
	}
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline Real* getArg1() { return fv; } typedef Real type1;inline const IndexInt& getArg2() { return N_row; } typedef IndexInt type2;inline const IndexInt& getArg3() { return off_begin; } typedef IndexInt type3;inline const Grid<Real> & getArg4() { return phi; } typedef Grid<Real>  type4;inline const Real& getArg5() { return scale; } typedef Real type5;inline const ParticleDataImpl<int> * getArg6() { return ptype; } typedef ParticleDataImpl<int>  type6;inline const int& getArg7() { return exclude; } typedef int type7;inline const int& getArg8() { return window; } typedef int type8; void runMessage() { debMsg("Executing kernel knExtractFeaturePhi ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,fv,N_row,off_begin,phi,scale,ptype,exclude,window);  }   } const BasicParticleSystem& p; Real* fv; const IndexInt N_row; const IndexInt off_begin; const Grid<Real> & phi; const Real scale; const ParticleDataImpl<int> * ptype; const int exclude; const int window;   };
#line 75 "plugin/tfplugins.cpp"






 struct knExtractFeatureGeo : public KernelBase { knExtractFeatureGeo( const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const FlagGrid &geo, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window) :  KernelBase(p.size()) ,p(p),fv(fv),N_row(N_row),off_begin(off_begin),geo(geo),scale(scale),ptype(ptype),exclude(exclude),window(window)   { runMessage(); run(); }   inline void op(IndexInt idx,  const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin, const FlagGrid &geo, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude, const int window )  {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (geo.is3D()) ? -window : 0, K = (geo.is3D()) ? window : 0;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k));
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Real geo_s = static_cast<Real>(geo.getAt(pos_s))*scale;
				const IndexInt off_geo = off_idx + off_begin + off_stencil;
				fv[off_geo] = geo_s;

				++off_stencil;
			}
		}
	}
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline Real* getArg1() { return fv; } typedef Real type1;inline const IndexInt& getArg2() { return N_row; } typedef IndexInt type2;inline const IndexInt& getArg3() { return off_begin; } typedef IndexInt type3;inline const FlagGrid& getArg4() { return geo; } typedef FlagGrid type4;inline const Real& getArg5() { return scale; } typedef Real type5;inline const ParticleDataImpl<int> * getArg6() { return ptype; } typedef ParticleDataImpl<int>  type6;inline const int& getArg7() { return exclude; } typedef int type7;inline const int& getArg8() { return window; } typedef int type8; void runMessage() { debMsg("Executing kernel knExtractFeatureGeo ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,fv,N_row,off_begin,geo,scale,ptype,exclude,window);  }   } const BasicParticleSystem& p; Real* fv; const IndexInt N_row; const IndexInt off_begin; const FlagGrid& geo; const Real scale; const ParticleDataImpl<int> * ptype; const int exclude; const int window;   };
#line 101 "plugin/tfplugins.cpp"







void extractFeatureVel(PyArrayContainer fv, const int N_row, const int off_begin, const BasicParticleSystem &p, const MACGrid &vel, const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0, const int window=1) {
	knExtractFeatureVel(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, vel, scale, ptype, exclude, window);
} static PyObject* _W_1 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "extractFeatureVel" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; PyArrayContainer fv = _args.get<PyArrayContainer >("fv",0,&_lock); const int N_row = _args.get<int >("N_row",1,&_lock); const int off_begin = _args.get<int >("off_begin",2,&_lock); const BasicParticleSystem& p = *_args.getPtr<BasicParticleSystem >("p",3,&_lock); const MACGrid& vel = *_args.getPtr<MACGrid >("vel",4,&_lock); const Real scale = _args.getOpt<Real >("scale",5,1.0,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtrOpt<ParticleDataImpl<int>  >("ptype",6,NULL,&_lock); const int exclude = _args.getOpt<int >("exclude",7,0,&_lock); const int window = _args.getOpt<int >("window",8,1,&_lock);   _retval = getPyNone(); extractFeatureVel(fv,N_row,off_begin,p,vel,scale,ptype,exclude,window);  _args.check(); } pbFinalizePlugin(parent,"extractFeatureVel", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("extractFeatureVel",e.what()); return 0; } } static const Pb::Register _RP_extractFeatureVel ("","extractFeatureVel",_W_1);  extern "C" { void PbRegister_extractFeatureVel() { KEEP_UNUSED(_RP_extractFeatureVel); } } 




void extractFeaturePhi(PyArrayContainer fv, const int N_row, const int off_begin, const BasicParticleSystem &p, const Grid<Real> &phi, const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0, const int window=1) {
	knExtractFeaturePhi(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, phi, scale, ptype, exclude, window);
} static PyObject* _W_2 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "extractFeaturePhi" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; PyArrayContainer fv = _args.get<PyArrayContainer >("fv",0,&_lock); const int N_row = _args.get<int >("N_row",1,&_lock); const int off_begin = _args.get<int >("off_begin",2,&_lock); const BasicParticleSystem& p = *_args.getPtr<BasicParticleSystem >("p",3,&_lock); const Grid<Real> & phi = *_args.getPtr<Grid<Real>  >("phi",4,&_lock); const Real scale = _args.getOpt<Real >("scale",5,1.0,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtrOpt<ParticleDataImpl<int>  >("ptype",6,NULL,&_lock); const int exclude = _args.getOpt<int >("exclude",7,0,&_lock); const int window = _args.getOpt<int >("window",8,1,&_lock);   _retval = getPyNone(); extractFeaturePhi(fv,N_row,off_begin,p,phi,scale,ptype,exclude,window);  _args.check(); } pbFinalizePlugin(parent,"extractFeaturePhi", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("extractFeaturePhi",e.what()); return 0; } } static const Pb::Register _RP_extractFeaturePhi ("","extractFeaturePhi",_W_2);  extern "C" { void PbRegister_extractFeaturePhi() { KEEP_UNUSED(_RP_extractFeaturePhi); } } 




void extractFeatureGeo(PyArrayContainer fv, const int N_row, const int off_begin, const BasicParticleSystem &p, const FlagGrid &flag, const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0, const int window=1) {
	knExtractFeatureGeo(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, flag, scale, ptype, exclude, window);
} static PyObject* _W_3 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "extractFeatureGeo" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; PyArrayContainer fv = _args.get<PyArrayContainer >("fv",0,&_lock); const int N_row = _args.get<int >("N_row",1,&_lock); const int off_begin = _args.get<int >("off_begin",2,&_lock); const BasicParticleSystem& p = *_args.getPtr<BasicParticleSystem >("p",3,&_lock); const FlagGrid& flag = *_args.getPtr<FlagGrid >("flag",4,&_lock); const Real scale = _args.getOpt<Real >("scale",5,1.0,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtrOpt<ParticleDataImpl<int>  >("ptype",6,NULL,&_lock); const int exclude = _args.getOpt<int >("exclude",7,0,&_lock); const int window = _args.getOpt<int >("window",8,1,&_lock);   _retval = getPyNone(); extractFeatureGeo(fv,N_row,off_begin,p,flag,scale,ptype,exclude,window);  _args.check(); } pbFinalizePlugin(parent,"extractFeatureGeo", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("extractFeatureGeo",e.what()); return 0; } } static const Pb::Register _RP_extractFeatureGeo ("","extractFeatureGeo",_W_3);  extern "C" { void PbRegister_extractFeatureGeo() { KEEP_UNUSED(_RP_extractFeatureGeo); } } 




// non-numpy related helpers

//! region detection functions

void floodFillRegion(Grid<int> &r, const FlagGrid &flags, const IndexInt idx, const int c, const int type) {
	r(idx) = c;
	if((flags(idx-flags.getStrideX()) & type) && !r[idx-flags.getStrideX()]) floodFillRegion(r, flags, idx-flags.getStrideX(), c, type);
	if((flags(idx+flags.getStrideX()) & type) && !r[idx+flags.getStrideX()]) floodFillRegion(r, flags, idx+flags.getStrideX(), c, type);
	if((flags(idx-flags.getStrideY()) & type) && !r[idx-flags.getStrideY()]) floodFillRegion(r, flags, idx-flags.getStrideY(), c, type);
	if((flags(idx+flags.getStrideY()) & type) && !r[idx+flags.getStrideY()]) floodFillRegion(r, flags, idx+flags.getStrideY(), c, type);
	if(!flags.is3D()) return;
	if((flags(idx-flags.getStrideZ()) & type) && !r[idx-flags.getStrideZ()]) floodFillRegion(r, flags, idx-flags.getStrideZ(), c, type);
	if((flags(idx+flags.getStrideZ()) & type) && !r[idx+flags.getStrideZ()]) floodFillRegion(r, flags, idx+flags.getStrideZ(), c, type);
}

int getRegions(Grid<int> &r, const FlagGrid &flags, const int ctype) {
	r.clear();
	int n_regions = 0;

	FOR_IDX(flags) {
		if((flags(idx) & ctype) && !r(idx)) floodFillRegion(r, flags, idx, ++n_regions, ctype);
	}
	return n_regions;
} static PyObject* _W_4 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "getRegions" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; Grid<int> & r = *_args.getPtr<Grid<int>  >("r",0,&_lock); const FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",1,&_lock); const int ctype = _args.get<int >("ctype",2,&_lock);   _retval = toPy(getRegions(r,flags,ctype));  _args.check(); } pbFinalizePlugin(parent,"getRegions", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("getRegions",e.what()); return 0; } } static const Pb::Register _RP_getRegions ("","getRegions",_W_4);  extern "C" { void PbRegister_getRegions() { KEEP_UNUSED(_RP_getRegions); } } 

void getRegionalCounts(Grid<int> &r, const FlagGrid &flags, const int ctype) {
	const int n_regions = getRegions(r, flags, ctype);
	std::vector<int> cnt(n_regions+1, 0);
	FOR_IDX(flags) {
		if(r[idx]>0) ++(cnt[r[idx]]);
	}
	FOR_IDX(flags) {
		r[idx] = cnt[r[idx]];
	}
} static PyObject* _W_5 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "getRegionalCounts" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; Grid<int> & r = *_args.getPtr<Grid<int>  >("r",0,&_lock); const FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",1,&_lock); const int ctype = _args.get<int >("ctype",2,&_lock);   _retval = getPyNone(); getRegionalCounts(r,flags,ctype);  _args.check(); } pbFinalizePlugin(parent,"getRegionalCounts", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("getRegionalCounts",e.what()); return 0; } } static const Pb::Register _RP_getRegionalCounts ("","getRegionalCounts",_W_5);  extern "C" { void PbRegister_getRegionalCounts() { KEEP_UNUSED(_RP_getRegionalCounts); } } 

void extendRegion(FlagGrid &flags, const int region, const int exclude, const int depth) {
	const int I=flags.getSizeX()-1, J=flags.getSizeY()-1, K=flags.getSizeZ()-1;
	for(int i_depth=0; i_depth<depth; ++i_depth) {
		std::vector<int> update;
		FOR_IJK(flags) {
			if(flags.get(i, j, k) & exclude) continue;
			if((i>0 && (flags.get(i-1, j, k)&region)) || (i<I && (flags.get(i+1, j, k)&region)) ||
			   (j>0 && (flags.get(i, j-1, k)&region)) || (j<J && (flags.get(i, j+1, k)&region)) ||
			   (flags.is3D() && ((k>0 && (flags.get(i, j, k-1)&region)) || (k<K && (flags.get(i, j, k+1)&region)))))
				update.push_back(flags.index(i, j, k));
		}

		for(std::vector<int>::const_iterator it=update.begin(); it!=update.end(); ++it) {
			flags[*it] = region;
		}
	}
} static PyObject* _W_6 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "extendRegion" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",0,&_lock); const int region = _args.get<int >("region",1,&_lock); const int exclude = _args.get<int >("exclude",2,&_lock); const int depth = _args.get<int >("depth",3,&_lock);   _retval = getPyNone(); extendRegion(flags,region,exclude,depth);  _args.check(); } pbFinalizePlugin(parent,"extendRegion", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("extendRegion",e.what()); return 0; } } static const Pb::Register _RP_extendRegion ("","extendRegion",_W_6);  extern "C" { void PbRegister_extendRegion() { KEEP_UNUSED(_RP_extendRegion); } } 

bool isIsolatedFluidCell(const IndexInt idx, const FlagGrid &flags) {
	if(!flags.isFluid(idx)) return false;
	if(flags.isFluid(idx-flags.getStrideX())) return false;
	if(flags.isFluid(idx+flags.getStrideX())) return false;
	if(flags.isFluid(idx-flags.getStrideY())) return false;
	if(flags.isFluid(idx+flags.getStrideY())) return false;
	if(!flags.is3D()) return true;
	if(flags.isFluid(idx-flags.getStrideZ())) return false;
	if(flags.isFluid(idx+flags.getStrideZ())) return false;
	return true;
}


 struct knMarkIsolatedFluidCell : public KernelBase { knMarkIsolatedFluidCell(FlagGrid &flags, const int mark) :  KernelBase(&flags,0) ,flags(flags),mark(mark)   { runMessage(); run(); }   inline void op(IndexInt idx, FlagGrid &flags, const int mark )  {
	if(isIsolatedFluidCell(idx, flags)) flags[idx] = mark;
}    inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline const int& getArg1() { return mark; } typedef int type1; void runMessage() { debMsg("Executing kernel knMarkIsolatedFluidCell ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,flags,mark);  }   } FlagGrid& flags; const int mark;   };
#line 216 "plugin/tfplugins.cpp"




void markIsolatedFluidCell(FlagGrid &flags, const int mark) {
	knMarkIsolatedFluidCell(flags, mark);
} static PyObject* _W_7 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "markIsolatedFluidCell" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",0,&_lock); const int mark = _args.get<int >("mark",1,&_lock);   _retval = getPyNone(); markIsolatedFluidCell(flags,mark);  _args.check(); } pbFinalizePlugin(parent,"markIsolatedFluidCell", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("markIsolatedFluidCell",e.what()); return 0; } } static const Pb::Register _RP_markIsolatedFluidCell ("","markIsolatedFluidCell",_W_7);  extern "C" { void PbRegister_markIsolatedFluidCell() { KEEP_UNUSED(_RP_markIsolatedFluidCell); } } 


 struct knMarkSmallRegions : public KernelBase { knMarkSmallRegions(FlagGrid &flags, const Grid<int> &rcnt, const int mark, const int exclude, const int th) :  KernelBase(&flags,0) ,flags(flags),rcnt(rcnt),mark(mark),exclude(exclude),th(th)   { runMessage(); run(); }   inline void op(IndexInt idx, FlagGrid &flags, const Grid<int> &rcnt, const int mark, const int exclude, const int th )  {
	if(flags[idx] & exclude) return;
	if(rcnt[idx] <= th) flags[idx] = mark;
}    inline FlagGrid& getArg0() { return flags; } typedef FlagGrid type0;inline const Grid<int> & getArg1() { return rcnt; } typedef Grid<int>  type1;inline const int& getArg2() { return mark; } typedef int type2;inline const int& getArg3() { return exclude; } typedef int type3;inline const int& getArg4() { return th; } typedef int type4; void runMessage() { debMsg("Executing kernel knMarkSmallRegions ", 3); debMsg("Kernel range" <<  " x "<<  maxX  << " y "<< maxY  << " z "<< minZ<<" - "<< maxZ  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,flags,rcnt,mark,exclude,th);  }   } FlagGrid& flags; const Grid<int> & rcnt; const int mark; const int exclude; const int th;   };
#line 226 "plugin/tfplugins.cpp"




void markSmallRegions(FlagGrid &flags, const Grid<int> &rcnt, const int mark, const int exclude, const int th=1) {
	knMarkSmallRegions(flags, rcnt, mark, exclude, th);
} static PyObject* _W_8 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "markSmallRegions" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",0,&_lock); const Grid<int> & rcnt = *_args.getPtr<Grid<int>  >("rcnt",1,&_lock); const int mark = _args.get<int >("mark",2,&_lock); const int exclude = _args.get<int >("exclude",3,&_lock); const int th = _args.getOpt<int >("th",4,1,&_lock);   _retval = getPyNone(); markSmallRegions(flags,rcnt,mark,exclude,th);  _args.check(); } pbFinalizePlugin(parent,"markSmallRegions", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("markSmallRegions",e.what()); return 0; } } static const Pb::Register _RP_markSmallRegions ("","markSmallRegions",_W_8);  extern "C" { void PbRegister_markSmallRegions() { KEEP_UNUSED(_RP_markSmallRegions); } } 

// particle helpers


 struct KnAddForcePvel : public KernelBase { KnAddForcePvel(ParticleDataImpl<Vec3> &v, const Vec3 &da, const ParticleDataImpl<int> *ptype, const int exclude) :  KernelBase(v.size()) ,v(v),da(da),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, ParticleDataImpl<Vec3> &v, const Vec3 &da, const ParticleDataImpl<int> *ptype, const int exclude )  {
	if(ptype && ((*ptype)[idx] & exclude)) return;
	v[idx] += da;
}    inline ParticleDataImpl<Vec3> & getArg0() { return v; } typedef ParticleDataImpl<Vec3>  type0;inline const Vec3& getArg1() { return da; } typedef Vec3 type1;inline const ParticleDataImpl<int> * getArg2() { return ptype; } typedef ParticleDataImpl<int>  type2;inline const int& getArg3() { return exclude; } typedef int type3; void runMessage() { debMsg("Executing kernel KnAddForcePvel ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,v,da,ptype,exclude);  }   } ParticleDataImpl<Vec3> & v; const Vec3& da; const ParticleDataImpl<int> * ptype; const int exclude;   };
#line 239 "plugin/tfplugins.cpp"

 
//! add force to vec3 particle data (ie, a velocity)
void addForcePvel(ParticleDataImpl<Vec3> &vel, const Vec3 &a, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude) {
	KnAddForcePvel(vel, a*dt, ptype, exclude);
} static PyObject* _W_9 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "addForcePvel" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; ParticleDataImpl<Vec3> & vel = *_args.getPtr<ParticleDataImpl<Vec3>  >("vel",0,&_lock); const Vec3& a = _args.get<Vec3 >("a",1,&_lock); const Real dt = _args.get<Real >("dt",2,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtr<ParticleDataImpl<int>  >("ptype",3,&_lock); const int exclude = _args.get<int >("exclude",4,&_lock);   _retval = getPyNone(); addForcePvel(vel,a,dt,ptype,exclude);  _args.check(); } pbFinalizePlugin(parent,"addForcePvel", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("addForcePvel",e.what()); return 0; } } static const Pb::Register _RP_addForcePvel ("","addForcePvel",_W_9);  extern "C" { void PbRegister_addForcePvel() { KEEP_UNUSED(_RP_addForcePvel); } } 


 struct KnUpdateVelocityFromDeltaPos : public KernelBase { KnUpdateVelocityFromDeltaPos(const BasicParticleSystem &p, ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &x_prev, const Real over_dt, const ParticleDataImpl<int> *ptype, const int exclude) :  KernelBase(p.size()) ,p(p),v(v),x_prev(x_prev),over_dt(over_dt),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, const BasicParticleSystem &p, ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &x_prev, const Real over_dt, const ParticleDataImpl<int> *ptype, const int exclude )  {
	if(ptype && ((*ptype)[idx] & exclude)) return;
	v[idx] = (p[idx].pos - x_prev[idx])*over_dt;
}    inline const BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline ParticleDataImpl<Vec3> & getArg1() { return v; } typedef ParticleDataImpl<Vec3>  type1;inline const ParticleDataImpl<Vec3> & getArg2() { return x_prev; } typedef ParticleDataImpl<Vec3>  type2;inline const Real& getArg3() { return over_dt; } typedef Real type3;inline const ParticleDataImpl<int> * getArg4() { return ptype; } typedef ParticleDataImpl<int>  type4;inline const int& getArg5() { return exclude; } typedef int type5; void runMessage() { debMsg("Executing kernel KnUpdateVelocityFromDeltaPos ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,v,x_prev,over_dt,ptype,exclude);  }   } const BasicParticleSystem& p; ParticleDataImpl<Vec3> & v; const ParticleDataImpl<Vec3> & x_prev; const Real over_dt; const ParticleDataImpl<int> * ptype; const int exclude;   };
#line 249 "plugin/tfplugins.cpp"

 
//! retrieve velocity from position change
void updateVelocityFromDeltaPos(BasicParticleSystem& parts, ParticleDataImpl<Vec3> &vel, const ParticleDataImpl<Vec3> &x_prev, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude) {
	KnUpdateVelocityFromDeltaPos(parts, vel, x_prev, 1.0/dt, ptype, exclude);
} static PyObject* _W_10 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "updateVelocityFromDeltaPos" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; BasicParticleSystem& parts = *_args.getPtr<BasicParticleSystem >("parts",0,&_lock); ParticleDataImpl<Vec3> & vel = *_args.getPtr<ParticleDataImpl<Vec3>  >("vel",1,&_lock); const ParticleDataImpl<Vec3> & x_prev = *_args.getPtr<ParticleDataImpl<Vec3>  >("x_prev",2,&_lock); const Real dt = _args.get<Real >("dt",3,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtr<ParticleDataImpl<int>  >("ptype",4,&_lock); const int exclude = _args.get<int >("exclude",5,&_lock);   _retval = getPyNone(); updateVelocityFromDeltaPos(parts,vel,x_prev,dt,ptype,exclude);  _args.check(); } pbFinalizePlugin(parent,"updateVelocityFromDeltaPos", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("updateVelocityFromDeltaPos",e.what()); return 0; } } static const Pb::Register _RP_updateVelocityFromDeltaPos ("","updateVelocityFromDeltaPos",_W_10);  extern "C" { void PbRegister_updateVelocityFromDeltaPos() { KEEP_UNUSED(_RP_updateVelocityFromDeltaPos); } } 

//! simple foward Euler integration for particle system

 struct KnStepEuler : public KernelBase { KnStepEuler(BasicParticleSystem &p, const ParticleDataImpl<Vec3> &v, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude) :  KernelBase(p.size()) ,p(p),v(v),dt(dt),ptype(ptype),exclude(exclude)   { runMessage(); run(); }   inline void op(IndexInt idx, BasicParticleSystem &p, const ParticleDataImpl<Vec3> &v, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude )  {
	if(ptype && ((*ptype)[idx] & exclude)) return;
	p[idx].pos += v[idx]*dt;
}    inline BasicParticleSystem& getArg0() { return p; } typedef BasicParticleSystem type0;inline const ParticleDataImpl<Vec3> & getArg1() { return v; } typedef ParticleDataImpl<Vec3>  type1;inline const Real& getArg2() { return dt; } typedef Real type2;inline const ParticleDataImpl<int> * getArg3() { return ptype; } typedef ParticleDataImpl<int>  type3;inline const int& getArg4() { return exclude; } typedef int type4; void runMessage() { debMsg("Executing kernel KnStepEuler ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,p,v,dt,ptype,exclude);  }   } BasicParticleSystem& p; const ParticleDataImpl<Vec3> & v; const Real dt; const ParticleDataImpl<int> * ptype; const int exclude;   };
#line 260 "plugin/tfplugins.cpp"


void eulerStep(BasicParticleSystem& parts, const ParticleDataImpl<Vec3> &vel, const ParticleDataImpl<int> *ptype, const int exclude) {
	KnStepEuler(parts, vel, parts.getParent()->getDt(), ptype, exclude);
} static PyObject* _W_11 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "eulerStep" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; BasicParticleSystem& parts = *_args.getPtr<BasicParticleSystem >("parts",0,&_lock); const ParticleDataImpl<Vec3> & vel = *_args.getPtr<ParticleDataImpl<Vec3>  >("vel",1,&_lock); const ParticleDataImpl<int> * ptype = _args.getPtr<ParticleDataImpl<int>  >("ptype",2,&_lock); const int exclude = _args.get<int >("exclude",3,&_lock);   _retval = getPyNone(); eulerStep(parts,vel,ptype,exclude);  _args.check(); } pbFinalizePlugin(parent,"eulerStep", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("eulerStep",e.what()); return 0; } } static const Pb::Register _RP_eulerStep ("","eulerStep",_W_11);  extern "C" { void PbRegister_eulerStep() { KEEP_UNUSED(_RP_eulerStep); } } 



 struct KnSetType : public KernelBase { KnSetType(ParticleDataImpl<int> &ptype, BasicParticleSystem &part, const int mark, const int stype, const FlagGrid &flags, const int cflag) :  KernelBase(ptype.size()) ,ptype(ptype),part(part),mark(mark),stype(stype),flags(flags),cflag(cflag)   { runMessage(); run(); }   inline void op(IndexInt idx, ParticleDataImpl<int> &ptype, BasicParticleSystem &part, const int mark, const int stype, const FlagGrid &flags, const int cflag )  {
	if(flags.isInBounds(part.getPos(idx), 0) && (flags.getAt(part.getPos(idx))&cflag) && (ptype[idx]&stype)) ptype[idx] = mark;
}    inline ParticleDataImpl<int> & getArg0() { return ptype; } typedef ParticleDataImpl<int>  type0;inline BasicParticleSystem& getArg1() { return part; } typedef BasicParticleSystem type1;inline const int& getArg2() { return mark; } typedef int type2;inline const int& getArg3() { return stype; } typedef int type3;inline const FlagGrid& getArg4() { return flags; } typedef FlagGrid type4;inline const int& getArg5() { return cflag; } typedef int type5; void runMessage() { debMsg("Executing kernel KnSetType ", 3); debMsg("Kernel range" <<  " size "<<  size  << " "   , 4); }; void run() {   const IndexInt _sz = size; 
#pragma omp parallel 
 {  
#pragma omp for  
  for (IndexInt i = 0; i < _sz; i++) op(i,ptype,part,mark,stype,flags,cflag);  }   } ParticleDataImpl<int> & ptype; BasicParticleSystem& part; const int mark; const int stype; const FlagGrid& flags; const int cflag;   };
#line 270 "plugin/tfplugins.cpp"



void setPartType(BasicParticleSystem &parts, ParticleDataImpl<int> &ptype, const int mark, const int stype, const FlagGrid &flags, const int cflag) {
	KnSetType(ptype, parts, mark, stype, flags, cflag);
} static PyObject* _W_12 (PyObject* _self, PyObject* _linargs, PyObject* _kwds) { try { PbArgs _args(_linargs, _kwds); FluidSolver *parent = _args.obtainParent(); bool noTiming = _args.getOpt<bool>("notiming", -1, 0); pbPreparePlugin(parent, "setPartType" , !noTiming ); PyObject *_retval = 0; { ArgLocker _lock; BasicParticleSystem& parts = *_args.getPtr<BasicParticleSystem >("parts",0,&_lock); ParticleDataImpl<int> & ptype = *_args.getPtr<ParticleDataImpl<int>  >("ptype",1,&_lock); const int mark = _args.get<int >("mark",2,&_lock); const int stype = _args.get<int >("stype",3,&_lock); const FlagGrid& flags = *_args.getPtr<FlagGrid >("flags",4,&_lock); const int cflag = _args.get<int >("cflag",5,&_lock);   _retval = getPyNone(); setPartType(parts,ptype,mark,stype,flags,cflag);  _args.check(); } pbFinalizePlugin(parent,"setPartType", !noTiming ); return _retval; } catch(std::exception& e) { pbSetError("setPartType",e.what()); return 0; } } static const Pb::Register _RP_setPartType ("","setPartType",_W_12);  extern "C" { void PbRegister_setPartType() { KEEP_UNUSED(_RP_setPartType); } } 


} //namespace


