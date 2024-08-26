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
#include "ibex_Strategy.h"
#include "ibex_CellBeamSearch.h"

namespace ibex {

class Bandit : public Strategy {

private:

    std::vector<std::vector<double>> matrizQ; // Matriz de 2D

public:

    //constructor
    Bandit(CellBeamSearch * buffer, int num_actions, double size_step);

    ~Bandit();

    void modeTraining();

    void resetVars();

    double calculateRewardExploration();

    double calculateRewardExplotation();

    int selectAction(int state);

    void MonitoringSize();

    void MonitoringChange();

    void printQ();

    void adder(bool loupChanged);

    double generateRandomDouble();

    int generateRandomInt(int range);

    void saveVectorsToFile();

    void loadVectorsFromFile();

    void StartExploration();
        
    void StartExplotation();

    int getActualState();

    int getActualAction();

    int getPastState();

    int getPastAction();

    void saveLogs();

    void updateQ(int actual_state, int actual_action, double reward);
    void updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward);

    int ActualState(bool searchType, double width, int activeNodes, bool loupChange);
    int ActualState(bool searchType);

    int getWidthCategory(double w);

    int getActiveNodesCategory(int an);

    void setLoupChanged(bool loupChange);

    void updateWidth(double loup, double uplo);

};

} // end namespace ibex

#endif //__IBEX__BANDIT_H__