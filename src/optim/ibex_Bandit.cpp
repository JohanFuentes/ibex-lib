#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdlib>  // Para rand() y srand()
#include <ctime>    // Para time()
#include <fstream>
#include <sstream>
#include <string>
#include "ibex_Bandit.h"

namespace ibex {

//constructor
Bandit::Bandit(CellBeamSearch * buffer, int num_actions, double size_step) : buffer(buffer), k(num_actions), alpha(size_step), matrizQ(2, std::vector<double>(num_actions, 0.0)) {
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
}

Bandit::~Bandit(){
    if(buffer != nullptr){
        delete buffer;
    }
}

void Bandit::modeTraining(){
    training = true;
}

void Bandit::resetVars(){
    nb_cells = 0;
    loup_found = 1;
}

double Bandit::calculateRewardExploration(){
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

double Bandit::calculateRewardExplotation(){

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

//Cambiar funcion de costo
int Bandit::selectAction(int state){

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
        
        // Selección greedy
        std::vector<int> max_indices;
        double max_value = *std::max_element(matrizQ[state].begin(), matrizQ[state].end());
        //std::cout<<"Greedy"<<std::endl;

        //Recorrer matrizQ[state]
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
        
        //std::cout<<"Action: "<<action<<std::endl;
        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    } else {
        //std::cout<<"Exploration"<<std::endl;
        // Selección aleatoria
        logStream << "Seleccion Aleatoria:" << std::endl;
        int randomAction = generateRandomInt(matrizQ[state].size());
        //std::cout<<randomAction<<std::endl;
        action = randomAction;
        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    }

    return action;
}

void Bandit::updateQ(int actual_state, int actual_action, double reward){

    //Recorrer matrizQ[state]
    logStream << "Q Antes:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }

    matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + alpha*(reward - matrizQ[actual_state][actual_action]);  
    //matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + alpha*(reward - matrizQ[actual_state][actual_action]);   
    logStream << std::endl<<"Q Despues:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }
}

void Bandit::MonitoringSize(){
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
            updateQ(estado_anterior, accion_anterior, reward);
        }

        estado_anterior = estado_actual;
        accion_anterior = accion_actual;
        
        estado_actual = ActualState(true);
        accion_actual = selectAction(estado_actual);
        
        if(training){
            reward = calculateRewardExplotation();
            logStream << "Update Q: " <<estado_anterior<<" , "<<accion_anterior<<std::endl;
            updateQ(estado_anterior, accion_anterior, reward);
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
    }
}

void Bandit::MonitoringChange(){

    if(change){
        logStream << "Monitoring Change (Explotacion):" << std::endl;
        
        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        estado_anterior = estado_actual;
        accion_anterior = accion_actual;
        
        estado_actual = ActualState(false);
        accion_actual = selectAction(estado_actual);

        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        buffer->setCost2Function(accion_actual);
        change = false;
    }
}

void Bandit::printQ(){
    for (const auto& fila : matrizQ) {
        for (int elemento : fila) {
            std::cout << elemento << "\t";  // Usamos tabulación para alinear los elementos
        }
        std::cout << std::endl;  // Nueva línea al final de cada fila
    }
}

void Bandit::adder(bool loupChanged){
    if(training){
        nb_cells += 2;
        if(loupChanged){loup_found += 1;}
    }
}

double Bandit::generateRandomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int Bandit::generateRandomInt(int range) {
    return rand() % range;
}

void Bandit::saveVectorsToFile(){

    if(training){
        // Abrir un archivo de texto para escribir
        std::ofstream file("MatrixQBandit.txt");
        if (!file.is_open()) {
            //std::cerr << "Error al abrir el archivo para escritura." << std::endl;

        }else{
            // Escribir los elementos de la matriz en el archivo
            for (const auto& fila : matrizQ) {
                for (int elemento : fila) {
                    file << elemento << " ";  // Separar cada elemento con un espacio
                }
                file << "\n";  // Nueva línea al final de cada fila
            }

            // Cerrar el archivo
            file.close();

            //std::cout << "Matriz guardada en 'MatrixQ.txt'." << std::endl;
        }

    }
}


void Bandit::loadVectorsFromFile() {
        std::ifstream file("MatrixQBandit.txt");
        if (!file.is_open()) {
            std::cerr << "Error al abrir el archivo para lectura." << std::endl;
        }else{
            std::string linea;
            int filaIndex = 0;
            while (getline(file, linea)) {
                std::istringstream iss(linea);
                double valor;
                int colIndex = 0;

                while (iss >> valor) {
                    if (filaIndex < matrizQ.size() && colIndex < matrizQ[filaIndex].size()) {
                        matrizQ[filaIndex][colIndex] = valor;  // Actualiza el valor existente
                    } else if (filaIndex < matrizQ.size()) {
                        matrizQ[filaIndex].push_back(valor);  // Añade un nuevo valor si la columna no existe
                    }
                    colIndex++;
                }
                filaIndex++;
            }

            file.close();
        }
}

int Bandit::ActualState(bool searchType){
    
    int state = -1;

    if (!searchType) {
        // Cuando searchType es False (Explotación)
        state = 0;
    } else {
        // Cuando searchType es True (Exploración)
        state = 1;
    }

    return state;

}

void Bandit::StartExploration() {

    loadVectorsFromFile();

    logStream << "Start Exploration:" << std::endl;
    logStream << "Estado anterior: " <<estado_anterior<<std::endl;
    logStream << "Accion anterior: " <<accion_anterior<<std::endl;
    logStream << "Estado actual: " <<estado_actual<<std::endl;
    logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

    estado_actual = ActualState(true); // Exploración(1)
    accion_actual = selectAction(estado_actual);
    
    logStream << "Se cambio estado actual y accion actual:" << std::endl;
    logStream << "Estado anterior: " <<estado_anterior<<std::endl;
    logStream << "Accion anterior: " <<accion_anterior<<std::endl;
    logStream << "Estado actual: " <<estado_actual<<std::endl;
    logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

    buffer->setCost2Function(accion_actual);
    start = true;
}

void Bandit::StartExplotation() {
    if(start){
        logStream << "Start Explotation:" << std::endl;
        estado_anterior = estado_actual;
        estado_actual = ActualState(false);
        accion_anterior = accion_actual;
        accion_actual = selectAction(estado_actual);

        logStream << "Se cambio estado actual, accion actual, estado anterior y accion anterior:" << std::endl;
        logStream << "Estado anterior: " <<estado_anterior<<std::endl;
        logStream << "Accion anterior: " <<accion_anterior<<std::endl;
        logStream << "Estado actual: " <<estado_actual<<std::endl;
        logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

        buffer->setCost2Function(accion_actual);
        start = false;
    }
}

int Bandit::getActualState(){
    return estado_actual;
}

int Bandit::getActualAction(){
    return accion_actual;
}

int Bandit::getPastState(){
    return estado_anterior;
}

int Bandit::getPastAction(){
    return accion_anterior;
}

void Bandit::saveLogs() {
    std::ofstream file("Logs.txt", std::ios::app);  // Abre el archivo en modo append
    if (file.is_open()) {
        file << logStream.str();  // Escribe los logs acumulados al archivo
        logStream.str("");  // Limpia el stream si deseas reutilizarlo
        logStream.clear();  // Limpia banderas de error
    } else {
        //std::cerr << "Failed to open the log file." << std::endl;
    }
}

} // end namespace ibex
