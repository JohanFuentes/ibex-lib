//============================================================================
//                                  I B E X                                   
// File        : Bisector
// Author      : Gilles Chabert
// Copyright   : Ecole des Mines de Nantes (France)
// License     : See the LICENSE file
// Created     : May 8, 2012
// Last Update : May 8, 2012
//============================================================================

#ifndef __IBEX_BISECTOR_H__
#define __IBEX_BISECTOR_H__

#include "ibex_Cell.h"

namespace ibex {

class Cell;

/** \ingroup bisector
 * \brief Generic bisector
 *
 * Base class of all bisectors. A bisector is an operator used by a strategy
 * (like \link ibex::Paver paver \endlink) to make a choice point.
 * An implementation of this class must provide a #bisect() function.
 *
 */
class Bsc {
public:
	/**
	 * \brief Bisect the current box and return the result.
	 */
	virtual std::pair<IntervalVector,IntervalVector> bisect(const IntervalVector& box)=0;

	/**
	 * \brief Bisect the current cell and return the result.
	 *
	 * The information in a cell is e.g., used to get the last bisected variable in case
	 * the bisector is called by a solver/paver (e.g., RoundRobin).
	 * Implementation is <b>optional</b>. By default, this function call bisect(cell.box).
	 * See #bisect(const IntervalVector&).
	 */
	virtual std::pair<Cell*,Cell*> bisect(Cell& cell);

	/**
	 * \brief Delete *this.
	 */
	virtual ~Bsc() { }

};

} // end namespace ibex
#endif // __IBEX_BISECTOR_H__
