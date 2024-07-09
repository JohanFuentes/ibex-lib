//============================================================================
//                                  I B E X                                   
// File        : ibex_CellBeamSearch.cpp
// Author      : Bertrand Neveu
// Copyright   : IMT Atlantique (France)
// License     : See the LICENSE file
// Created     : Aug 29, 2017
// Modified    : Sep 07, 2017
//============================================================================

#include "ibex_CellBeamSearch.h"

using namespace std;

namespace ibex {

CellBeamSearch::CellBeamSearch(CellDoubleHeap& currentbuffer, CellDoubleHeap& futurebuffer, const ExtendedSystem & sys, unsigned int beamsize) : CellDoubleHeap (sys), beamsize(beamsize), currentbuffer(currentbuffer), futurebuffer (futurebuffer) {

}

CellBeamSearch::~CellBeamSearch() {

}

bool CellBeamSearch::empty() const {
	return (currentbuffer.empty()  && futurebuffer.empty() && CellDoubleHeap::empty());
}

unsigned int  CellBeamSearch::size() const {
	return (currentbuffer.size()+ futurebuffer.size() + CellDoubleHeap::size());
}

void CellBeamSearch::flush() {
	currentbuffer.flush();
	futurebuffer.flush();
	CellDoubleHeap::flush();
}

void CellBeamSearch::contract(double new_loup) {

	if (!(CellDoubleHeap::empty()))	DoubleHeap<Cell>::contract(new_loup);

	if (!(currentbuffer.empty()))
		currentbuffer.contract(new_loup);

	if (!(futurebuffer.empty())) {
		futurebuffer.contract(new_loup);

	}
}

void CellBeamSearch::push(Cell* cell) {
	futurebuffer.push(cell);
}

/*
double CellBeamSearch::cell_cost(const Cell& cell) const {
	return cell.box[sys.goal_var()].lb(); // lb linea original 
	//return cell.box[sys.goal_var()].ub(); // ub
}
*/

// returns the cell to handled
Cell* CellBeamSearch::pop() {
	if (! (currentbuffer.empty()) )
		return currentbuffer.pop();
	else if (! (futurebuffer.empty()) ) {
		Cell * c= futurebuffer.pop();
		move_buffers();
		return c;
	}
	else return CellDoubleHeap::pop();
}

// emptying the futurebuffer : buffersize-1 cells are put into
// the currentbuffer , the remaining into the global heap
void CellBeamSearch::move_buffers() {
	while (! (futurebuffer.empty())) {
		if (currentbuffer.size() < beamsize-1)
			currentbuffer.push(futurebuffer.pop());
		else
			CellDoubleHeap::push(futurebuffer.pop());
	}
}

Cell* CellBeamSearch::top() const {
	if (! (currentbuffer.empty()) ) {
		return currentbuffer.top();
	}
	else
		if (! (futurebuffer.empty()) ) {
			return futurebuffer.top();
		}
		else {
			return CellDoubleHeap::top();
		}
}

// the minimum of all open nodes
double CellBeamSearch::minimum() const {
	assert (!(empty()));
	if  (! (currentbuffer.empty()) && !(futurebuffer.empty()) &&  (!CellDoubleHeap::empty())){
		//      cout << "minimum " << currentbuffer.minimum() << "  " << futurebuffer.minimum() <<  " " << CellHeap::minimum() << endl;
		return std::min (currentbuffer.minimum(), std::min( futurebuffer.minimum(),  CellDoubleHeap::minimum()));
	}
	else if (! (currentbuffer.empty()) && !CellDoubleHeap::empty())
		return std::min(currentbuffer.minimum(), CellDoubleHeap::minimum());
	else if (! (futurebuffer.empty()) && !CellDoubleHeap::empty())
		return std::min(futurebuffer.minimum(), CellDoubleHeap::minimum());
	else if  (! (futurebuffer.empty()) && !currentbuffer.empty())
		return std::min(futurebuffer.minimum(), currentbuffer.minimum());
	else if  (! (futurebuffer.empty()))
		return futurebuffer.minimum();
	else if  (!(currentbuffer.empty()))
		return currentbuffer.minimum();
	else
		return CellDoubleHeap::minimum();
}

} // end namespace
