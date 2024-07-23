#ifndef __IBEX__SIMPLE_BANDIT_H__
#define __IBEX__SIMPLE_BANDIT_H__

#include <iostream>
#include <vector>
#include <cmath>
//#include <random>
#include <algorithm>
#include <cstdlib>  // Para rand() y srand()
#include <ctime>    // Para time()
#include "ibex_CellBeamSearch.h"


namespace ibex {

class SimpleBandit {

private:

    CellBeamSearch * buffer;
    std::vector<double> Q_exploration;   // Estimaciones de recompensa para cada acción
    std::vector<double> Q_explotation;   // Estimaciones de recompensa para cada acción
    //std::vector<int> N;      // Contador de selecciones para cada acción
    double epsilon;          // Probabilidad de exploración
    double alpha;            // Tasa de aprendizaje
    int k;                   // Número de acciones
    bool change;
    int nb_cells;
    double loup_variation;
    int action_exploration;
    int action_explotation;
    bool update;
    //std::default_random_engine generator;
    //std::uniform_real_distribution<double> prob;
    //std::uniform_int_distribution<int> action_selector;

public:

    //constructor
    SimpleBandit(CellBeamSearch * buffer, int num_actions, double size_step);

    ~SimpleBandit();

/*
    void setLoupVariation(double variation);

    void setNbCells(int nb);
*/
    void resetVars();

    double calculateRewardExploration();

    double calculateRewardExplotation();

    void selectActionExploration();

    void selectActionExplotation();

    void updateQExploration();

    void updateQExplotation();

    void MonitoringSize();

    void MonitoringChange();

    void printQ();

    void adder(bool loupChanged);

    double generateRandomDouble();

    int generateRandomInt();

    void saveVectorsToFile();

    void loadVectorsFromFile();

};

} // end namespace ibex

#endif //__IBEX__SIMPLE_BANDIT_H__