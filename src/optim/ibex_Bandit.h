#ifndef __IBEX__BANDIT_H__
#define __IBEX__BANDIT_H__

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

class Bandit {

private:

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
    std::vector<std::vector<double>> matrizQ; // Matriz de 2D
    bool start; // Para saber si es la primera iteración del algoritmo
    bool training;

public:

    //constructor
    Bandit(CellBeamSearch * buffer, int num_actions, double size_step);

    ~Bandit();

    void modeTraining();

    void resetVars();

    double calculateRewardExploration();

    double calculateRewardExplotation();

    int selectAction(int state);

    void updateQ(int actual_state, int actual_action, double reward);

    void MonitoringSize();

    void MonitoringChange();

    void printQ();

    void adder(bool loupChanged);

    double generateRandomDouble();

    int generateRandomInt(int range);

    void saveVectorsToFile();

    void loadVectorsFromFile();

    int ActualState(bool searchType);

    void StartExploration();
        
    void StartExplotation();

    int getActualState();

    int getActualAction();

    int getPastState();

    int getPastAction();

    void saveLogs();

};

} // end namespace ibex

#endif //__IBEX__BANDIT_H__