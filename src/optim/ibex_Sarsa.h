#ifndef __IBEX__SARSA_H__
#define __IBEX__SARSA_H__

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

class Sarsa {

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
    bool loup_changed;
    double width;
    bool training;

public:

    //constructor
    Sarsa(CellBeamSearch * buffer, int num_actions, double size_step);

    ~Sarsa();

    void modeTraining();

    void resetVars();

    double calculateRewardExploration();

    double calculateRewardExplotation();

    int selectAction(int state);

    void updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward);

    void MonitoringSize();

    void MonitoringChange();

    void printQ();

    void adder(bool loupChanged);

    double generateRandomDouble();

    int generateRandomInt(int range);

    void saveVectorsToFile();

    void loadVectorsFromFile();

    int ActualState(bool searchType, double width, int activeNodes, bool loupChange);

    int getWidthCategory(double w);

    int getActiveNodesCategory(int an);

    void StartExploration();
        
    void StartExplotation();

    int getActualState();

    int getActualAction();

    int getPastState();

    int getPastAction();

    void saveLogs();

    void setLoupChanged(bool loupChange);

    void updateWidth(double loup, double uplo);

};

} // end namespace ibex

#endif //__IBEX__SARSA_H__