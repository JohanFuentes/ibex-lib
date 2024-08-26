#ifndef __IBEX__STRATEGY_H__
#define __IBEX__STRATEGY_H__

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>  // Para rand() y srand()
#include <ctime>    // Para time()
#include <sstream>
#include <fstream>
#include <string>
#include "ibex_CellBeamSearch.h"


namespace ibex {

class Strategy {

protected:

    CellBeamSearch * buffer;
    double epsilon;          // Probabilidad de exploración
    double alpha;            // Tasa de aprendizaje
    int k;                   // Número de acciones
    bool change;
    int nb_cells;
    int loup_found;
    int estado_anterior;
    int accion_anterior;
    int estado_actual;
    int accion_actual;
    std::ostringstream logStream;  // Para acumular los mensajes de log
    bool start; // Para saber si es la primera iteración del algoritmo
    bool training;

public:

    //constructor
    Strategy(CellBeamSearch * buffer, int num_actions, double size_step);

    virtual ~Strategy();

    virtual void modeTraining() = 0;

    virtual void resetVars() = 0;

    virtual double calculateRewardExploration() = 0;

    virtual double calculateRewardExplotation() = 0;

    virtual int selectAction(int state) = 0;

    virtual void MonitoringSize() = 0;

    virtual void MonitoringChange() = 0;

    virtual void printQ() = 0;

    virtual void adder(bool loupChanged) = 0;

    virtual double generateRandomDouble() = 0;

    virtual int generateRandomInt(int range) = 0;

    virtual void saveVectorsToFile() = 0;

    virtual void loadVectorsFromFile() = 0;

    virtual void StartExploration() = 0;
        
    virtual void StartExplotation() = 0;

    virtual int getActualState() = 0;

    virtual int getActualAction() = 0;

    virtual int getPastState() = 0;

    virtual int getPastAction() = 0;

    virtual void saveLogs() = 0;

    virtual void updateQ(int actual_state, int actual_action, double reward)=0;
    virtual void updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward) = 0;

    virtual int ActualState(bool searchType)=0;
    virtual int ActualState(bool searchType, double width, int activeNodes, bool loupChange) = 0;

    virtual int getWidthCategory(double w) = 0;

    virtual int getActiveNodesCategory(int an) = 0;

    virtual void setLoupChanged(bool loupChange) = 0;

    virtual void updateWidth(double loup, double uplo) = 0;
};

} // end namespace ibex

#endif //__IBEX__STRATEGY_H__