#include <iostream>
#include "ibex_Strategy.h"

namespace ibex {

Strategy::Strategy(CellBeamSearch * buffer, int num_actions, double size_step) : buffer(buffer), k(num_actions), alpha(size_step) {
    epsilon = 0.1;
    change = false;
    start = false;
    nb_cells = 0;
    loup_found = 1;
    estado_anterior=0;
    accion_anterior=0;
    estado_actual=0;
    accion_actual=0;
    srand(static_cast<unsigned>(time(0)));
    training = false;
    ruleta = true;
}

Strategy::~Strategy(){}

} // end namespace ibex
