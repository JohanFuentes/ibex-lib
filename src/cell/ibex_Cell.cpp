//============================================================================
//                                  I B E X                                   
// File        : ibex_Cell.cpp
// Author      : Gilles Chabert
// Copyright   : IMT Atlantique (France)
// License     : See the LICENSE file
// Created     : May 10, 2012
// Last Update : Jun 07, 2018
//============================================================================

#include "ibex_Cell.h"
#include "ibex_Bsc.h"
#include <limits.h>
#include "ibex_Bxp.h"
#include "ibex_Bxp.h"

using namespace std;

namespace ibex {

Cell::Cell(const IntervalVector& box, int var, unsigned int depth, int d) : box(box), prop(this->box), bisected_var(var), depth(depth), d(d) {

}

Cell::Cell(const Cell& e) : box(e.box), prop(this->box, e.prop), bisected_var(e.bisected_var), depth(e.depth), d(e.d) {

}

pair<Cell*,Cell*> Cell::bisect(const BisectionPoint& pt) const {

	Cell* cleft;
	Cell* cright;

	if (pt.rel_pos) {
		pair<IntervalVector,IntervalVector> boxes=box.bisect(pt.var,pt.pos);
		cleft = new Cell(boxes.first, pt.var, depth+1);
		cright = new Cell(boxes.second, pt.var, depth+1);
	} else {
		IntervalVector b1(box);
		IntervalVector b2(box);
		b1[pt.var]=Interval(box[pt.var].lb(), pt.pos);
		b2[pt.var]=Interval(pt.pos, box[pt.var].ub());
		cleft = new Cell(b1, pt.var, depth+1);
		cright = new Cell(b2, pt.var, depth+1);
	}

	prop.update_bisect(Bisection(box, pt, cleft->box, cright->box), cleft->prop, cright->prop);

	return pair<Cell*,Cell*>(cleft,cright);
}

Cell::~Cell() {

}

int Cell::get_d(){
	return d;
}

void Cell::set_d(int new_d){
	d = new_d;
}

std::ostream& operator<<(std::ostream& os, const Cell& c) {
	os << c.box;
	return os;
}

} // end namespace ibex
