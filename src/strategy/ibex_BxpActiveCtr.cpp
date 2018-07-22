//============================================================================
//                                  I B E X
// File        : ibex_BxpActiveCtr.cpp
// Author      : Gilles Chabert
// Copyright   : IMT Atlantique (France)
// License     : See the LICENSE file
// Created     : Jul 05, 2018
//============================================================================

#include "ibex_BxpActiveCtr.h"
#include "ibex_Id.h"

namespace ibex {

Map<long,false> BxpActiveCtr::ids;

void BxpActiveCtr::check() {
	if (!up2date) {
		assert(_active);
		// TODO: if the impact contains no variable involved in the constraint
		// we can avoid evaluation
		Domain y=ctr.right_hand_side();
		switch (y.dim.type()) {
		case Dim::SCALAR:       _active = !(ctr.f.eval(box).is_subset(y.i())); break;
		case Dim::ROW_VECTOR:
		case Dim::COL_VECTOR:   _active = !(ctr.f.eval_vector(box).is_subset(y.v())); break;
		case Dim::MATRIX:       _active = !(ctr.f.eval_matrix(box).is_subset(y.m())); break;
		}
	}
	up2date=true;
}

void BxpActiveCtr::update(const BoxEvent& e, const BoxProperties& prop) {
	if (_active || e.type!=BoxEvent::CONTRACT) {
		_active=true;
		up2date=false;
	}
}

long BxpActiveCtr::get_id(const NumConstraint& ctr) {
	try {
		return ids[ctr.id];
	} catch(Map<long,false>::NotFound&) {
		long new_id=next_id();
		ids.insert_new(ctr.id, new_id);
		return new_id;
	}
}

} // end namespace ibex
