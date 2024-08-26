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
#include "ibex_Sarsa.h"

namespace ibex {

Sarsa::Sarsa(CellBeamSearch * buffer, int num_actions, double size_step) : Strategy(buffer,num_actions,size_step),
                                                                           matrizQ(60, std::vector<double>(num_actions, 0.0)),
                                                                           loup_changed(false),
                                                                           width(0.0)
{}


Sarsa::~Sarsa(){}

void Sarsa::modeTraining(){
    training = true;
}

void Sarsa::updateWidth(double loup, double uplo){
    width = loup - uplo;
}

void Sarsa::setLoupChanged(bool loupChange){
    loup_changed = loupChange;
}

void Sarsa::resetVars(){
    nb_cells = 0;
    loup_found = 1;
}

double Sarsa::calculateRewardExploration(){
    logStream << "Calculo recompensa exploracion:" << std::endl;
    logStream << "Numero de celdas: " <<nb_cells<<std::endl;
    logStream << "loups encontrados: " <<loup_found<<std::endl;
    double reward = 0.0;

    if(loup_found == 1){
        reward = -2.0;
    }else{
        reward = 10000.0*loup_found;
    }

    logStream << "Recompensa: "<< reward << std::endl<<std::endl;

    return reward;
}

double Sarsa::calculateRewardExplotation(){

    logStream << "Calculo recompensa explotacion:" << std::endl;
    logStream << "Numero de celdas: " <<nb_cells<<std::endl;
    logStream << "loups encontrados: " <<loup_found<<std::endl;

    double reward = 0.0;

    if(loup_found == 1){
        reward = -nb_cells*1.0;
    }else{
        reward = 100000.0*loup_found/nb_cells;    
    }

    logStream << "Recompensa: "<< reward << std::endl<<std::endl;

    return reward;
}

int Sarsa::selectAction(int state){

    int action = -1;

    double randomNum = 0.0;

    if(training){
        randomNum = generateRandomDouble();
    }else{
        randomNum = 1.0;
    }

    logStream << "Ingreso a Seleccion de accion:" << std::endl;

    if (randomNum > epsilon) {
        
        logStream << "Seleccion Greedy:" << std::endl;
        
        // Greedy selection
        std::vector<int> max_indices;
        double max_value = *std::max_element(matrizQ[state].begin(), matrizQ[state].end());

        //Travel matrizQ[state]
        logStream << "Vector Q:" << std::endl;
        for (size_t i = 0; i < matrizQ[state].size(); ++i) {
            logStream << matrizQ[state][i] << " ";
        }

        for (size_t i = 0; i < matrizQ[state].size(); ++i) {
            if (matrizQ[state][i] == max_value) {
                max_indices.push_back(i);
            }
        }

        int randomAction = generateRandomInt(max_indices.size());

        action = max_indices[randomAction];
        
        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    } else {
        // Random selection
        logStream << "Seleccion Aleatoria:" << std::endl;
        int randomAction = generateRandomInt(matrizQ[state].size());

        action = randomAction;
        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    }

    return action;
}

void Sarsa::updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward){

    //Travel matrizQ[state]
    logStream << "Q Antes:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }

    matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + alpha*(reward + 0.9*matrizQ[future_state][future_action] - matrizQ[actual_state][actual_action]);  
    //matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + alpha*(reward - matrizQ[actual_state][actual_action]);   
    logStream << "Q Despues:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }
}

void Sarsa::updateQ(int actual_state, int actual_action, double reward){}

void Sarsa::MonitoringSize(){
    if(buffer->futurebuffer.size() == 0){

        logStream << "Monitoring Size (Exploracion):" << std::endl;

        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        double reward = 0;

        if(training){
            reward = calculateRewardExploration();
            logStream << "Update Q: " <<estado_anterior<<" , "<<accion_anterior<<std::endl;
            updateQ(estado_anterior, estado_actual, accion_anterior, accion_actual, reward);
        }



        estado_anterior = estado_actual;
        accion_anterior = accion_actual;
        
        estado_actual = ActualState(true, width, buffer->CellDoubleHeap::size() , loup_changed);
        accion_actual = selectAction(estado_actual);
        

        if(training){
            reward = calculateRewardExplotation();
            logStream << "Update Q: " <<estado_anterior<<" , "<<accion_anterior<<std::endl;
            updateQ(estado_anterior, estado_actual, accion_anterior, accion_actual, reward);
        }

        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;


        buffer->setCost2Function(accion_actual);
        
        if(training){
            resetVars();
        }

        change = true;
        loup_changed = false;
    }
}

void Sarsa::MonitoringChange(){

    if(change){
        logStream << "Monitoring Change (Explotacion):" << std::endl;
        
        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        estado_anterior = estado_actual;
        accion_anterior = accion_actual;
        
        estado_actual = ActualState(false, width, -1, loup_changed);
        accion_actual = selectAction(estado_actual);

        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        buffer->setCost2Function(accion_actual);
        change = false;
        loup_changed = false;
    }
}

void Sarsa::printQ(){
    for (const auto& fila : matrizQ) {
        for (int elemento : fila) {
            std::cout << elemento << "\t";
        }
        std::cout << std::endl;
    }
}

void Sarsa::adder(bool loupChanged){
    if(training){
        nb_cells += 2;
        if(loupChanged){loup_found += 1;}
    }
}

double Sarsa::generateRandomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int Sarsa::generateRandomInt(int range) {
    return rand() % range;
}

void Sarsa::saveVectorsToFile(){

    if(training){

        std::ofstream file("MatrixQSarsa.txt");
        if (!file.is_open()) {
            //std::cerr << "Error al abrir el archivo para escritura." << std::endl;

        }else{

            for (const auto& fila : matrizQ) {
                for (int elemento : fila) {
                    file << elemento << " ";
                }
                file << "\n";
            }

            file.close();

        }

    }
}

void Sarsa::loadVectorsFromFile() {
        std::ifstream file("MatrixQSarsa.txt");
        if (!file.is_open()) {
            //std::cerr << "Error al abrir el archivo para lectura." << std::endl;
        }else{
            std::string linea;
            int filaIndex = 0;
            while (getline(file, linea)) {
                std::istringstream iss(linea);
                double valor;
                int colIndex = 0;

                while (iss >> valor) {
                    if (filaIndex < matrizQ.size() && colIndex < matrizQ[filaIndex].size()) {
                        matrizQ[filaIndex][colIndex] = valor;
                    } else if (filaIndex < matrizQ.size()) {
                        matrizQ[filaIndex].push_back(valor);
                    }
                    colIndex++;
                }
                filaIndex++;
            }

            file.close();
        }
}


int Sarsa::getWidthCategory(double width) {
    if (width < 0.05) return 0;
    if (width < 1) return 1;
    if (width < 5) return 2;
    if (width < 15) return 3;
    return 4; // width >= 15
}

int Sarsa::getActiveNodesCategory(int activeNodes) {
    if (activeNodes < 10) return 0;
    if (activeNodes < 30) return 1;
    if (activeNodes < 60) return 2;
    if (activeNodes < 100) return 3;
    return 4; // activeNodes >= 100
}

int Sarsa::ActualState(bool searchType, double width, int activeNodes, bool loupChange){
    
    /*
    searchType: False, True
    width: [0 - 1[,[1,5[,[5,15[,[15,50[,[50,infinity[
    activeNodes: [0, 10[,[10,30[,[30,60[,[60,100[,[100,infinity[
    loupChange: False, True
    */
    int state = -1;
    int widthCategory = getWidthCategory(width);
    int activeNodesCategory = getActiveNodesCategory(activeNodes);

    //int index = -1; // Value not valid for default

    if (!searchType) {
        // When searchType it's False, considerate only width and loupChange (10 possible combinations)
        state = widthCategory * 2 + (loupChange ? 1 : 0);
    } else {
        // When searchType it's True, considerate all the variables (50 possible combinations)
        state = 10 + (widthCategory * 10 + activeNodesCategory * 2 + (loupChange ? 1 : 0));
    }

    return state;

}

int Sarsa::ActualState(bool searchType){return 0;}

void Sarsa::StartExploration() {
    loadVectorsFromFile();
    logStream << "Start Exploration:" << std::endl;
    logStream << "Estado anterior: " <<estado_anterior<<std::endl;
    logStream << "Accion anterior: " <<accion_anterior<<std::endl;
    logStream << "Estado actual: " <<estado_actual<<std::endl;
    logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

    estado_actual = ActualState(true, width, 1, loup_changed);
    accion_actual = selectAction(estado_actual);
    
    logStream << "Se cambio estado actual y accion actual:" << std::endl;
    logStream << "Estado anterior: " <<estado_anterior<<std::endl;
    logStream << "Accion anterior: " <<accion_anterior<<std::endl;
    logStream << "Estado actual: " <<estado_actual<<std::endl;
    logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

    buffer->setCost2Function(accion_actual);
    start = true;
}

void Sarsa::StartExplotation() {
    if(start){
        estado_anterior = estado_actual;
        estado_actual = ActualState(false, width, -1, loup_changed);
        accion_anterior = accion_actual;
        accion_actual = selectAction(estado_actual);

        logStream << "Se cambio estado actual, accion actual, estado anterior y accion anterior:" << std::endl;
        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        buffer->setCost2Function(accion_actual);
        start = false;
        loup_changed = false;
    }
}

int Sarsa::getActualState(){
    return estado_actual;
}

int Sarsa::getActualAction(){
    return accion_actual;
}

int Sarsa::getPastState(){
    return estado_anterior;
}

int Sarsa::getPastAction(){
    return accion_anterior;
}

void Sarsa::saveLogs() {
    std::ofstream file("Logs.txt", std::ios::app);
    if (file.is_open()) {
        file << logStream.str();
        logStream.str("");
        logStream.clear();
    } else {
        //std::cerr << "Failed to open the log file." << std::endl;
    }
}

} // end namespace ibex
