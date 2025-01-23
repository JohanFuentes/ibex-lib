#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include "ibex_RoundRobin.h"

namespace ibex {

RoundRobin::RoundRobin(CellBeamSearch * buffer, int num_actions, double size_step) : Strategy(buffer,num_actions,size_step),
                                                                            option(0)
{}

RoundRobin::~RoundRobin(){}

// Change the cost function

int RoundRobin::selectAction(int state){return 0;}

int RoundRobin::selectAction(){

    int action = option;

    if(option+1 == 8){
        option = 0;
    }else{
        option = option + 1;
    }

    return action;
}


void RoundRobin::MonitoringSize(){
    if(buffer->futurebuffer.size() == 0){

        accion_actual = selectAction();
        //std::cout<<"Accion actual (MZ): "<<accion_actual<<std::endl;
        buffer->setCost2Function(accion_actual);
        change = true;
    }
}

void RoundRobin::MonitoringChange(){

    if(change){

        accion_actual = selectAction();
        //std::cout<<"Accion actual (MC): "<<accion_actual<<std::endl;
        buffer->setCost2Function(accion_actual);
        change = false;
    }
}

void RoundRobin::StartExploration() {

    accion_actual = selectAction();
    //std::cout<<"Accion actual (SExploration): "<<accion_actual<<std::endl;
    buffer->setCost2Function(accion_actual);
    start = true;
}

void RoundRobin::StartExplotation() {
    if(start){

        accion_actual = selectAction();
        //std::cout<<"Accion actual (SExplotation): "<<accion_actual<<std::endl;
        buffer->setCost2Function(accion_actual);
        start = false;
    }
}

int RoundRobin::getActualState(){
    return estado_actual;
}

int RoundRobin::getActualAction(){
    return accion_actual;
}

int RoundRobin::getPastState(){
    return estado_anterior;
}

int RoundRobin::getPastAction(){
    return accion_anterior;
}

double RoundRobin::generateRandomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int RoundRobin::generateRandomInt(int range) {
    return rand() % range;
}

void RoundRobin::saveLogs() {}

int RoundRobin::getWidthCategory(double width) {return 0;}

int RoundRobin::getActiveNodesCategory(int activeNodes) {return 0;}

void RoundRobin::updateWidth(double loup, double uplo){}

void RoundRobin::setLoupChanged(bool loupChange){}

void RoundRobin::modeTraining(){}

void RoundRobin::resetVars(){}

double RoundRobin::calculateRewardExploration(){return 0.0;}

double RoundRobin::calculateRewardExplotation(){return 0.0;}

void RoundRobin::updateQ(int actual_state, int actual_action, double reward){}

void RoundRobin::updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward){}

void RoundRobin::printQ(){}

void RoundRobin::adder(bool loupChanged){}

void RoundRobin::saveVectorsToFile(){}

void RoundRobin::loadVectorsFromFile() {}

int RoundRobin::ActualState(bool searchType){return 0;}

int RoundRobin::ActualState(bool searchType, double width, int activeNodes, bool loupChange){return 0;}
} // end namespace ibex
