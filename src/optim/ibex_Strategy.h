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

    CellBeamSearch * buffer;        // where the cells are stored
    double epsilon;                 // Probability of exploration
    double alpha;                   // Learning rate
    int k;                          // Number of actions
    bool change;                    // To know if the type search has changed
    int nb_cells;                   // Number of cells created in the explotation search
    int loup_found;                 // Number of loups found in the explotation search
    int estado_anterior;            // Previous state
    int accion_anterior;            // Previous action
    int estado_actual;              // Current state
    int accion_actual;              // Current action
    std::ostringstream logStream;   // Log stream to save the logs
    bool start;                     // To know if the search has started
    bool training;                  // To know if the strategy is in training mode
    bool ruleta;                    // To know if the selection is based on roulette

public:

    // Constructor and destructor

    Strategy(CellBeamSearch * buffer, int num_actions, double size_step);

    virtual ~Strategy();

    // Virtual methods

    // Activate training mode

    virtual void modeTraining() = 0;

    // Reset the variables nb_cells and loup_found

    virtual void resetVars() = 0;

    // Calculate the reward for the exploration search

    virtual double calculateRewardExploration() = 0;

    // Calculate the reward for the explotation search

    virtual double calculateRewardExplotation() = 0;

    // Select the action to take in the current state

    virtual int selectAction(int state) = 0;

    // Monitoring the size of the future buffer (if empty or not), for knowing if the type search has changed (exploration or explotation)

    virtual void MonitoringSize() = 0;

    // Monitoring the change of the type search

    virtual void MonitoringChange() = 0;

    // Print the matrix Q

    virtual void printQ() = 0;

    // nb_cells and loup_found adder values

    virtual void adder(bool loupChanged) = 0;

    // Generate a random double number

    virtual double generateRandomDouble() = 0;

    // Generate a random integer number

    virtual int generateRandomInt(int range) = 0;

    // Save the matrix Q to a file

    virtual void saveVectorsToFile() = 0;

    // Load the matrix Q from a file

    virtual void loadVectorsFromFile() = 0;

    // Start the exploration search

    virtual void StartExploration() = 0;
        
    // Start the explotation search

    virtual void StartExplotation() = 0;

    virtual int getActualState() = 0;

    virtual int getActualAction() = 0;

    virtual int getPastState() = 0;

    virtual int getPastAction() = 0;

    virtual void saveLogs() = 0;

    // Update the matrix Q with the reward obtained

    virtual void updateQ(int actual_state, int actual_action, double reward)=0;
    virtual void updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward) = 0;

    // Get the actual state of the search

    virtual int ActualState(bool searchType)=0;
    virtual int ActualState(bool searchType, double width, int activeNodes, bool loupChange) = 0;

    virtual int getWidthCategory(double w) = 0;

    virtual int getActiveNodesCategory(int an) = 0;

    virtual void setLoupChanged(bool loupChange) = 0;

    virtual void updateWidth(double loup, double uplo) = 0;
};

} // end namespace ibex

#endif //__IBEX__STRATEGY_H__