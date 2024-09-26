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
#include "ibex_Bandit.h"

namespace ibex {

Bandit::Bandit(CellBeamSearch * buffer, int num_actions, double size_step) : Strategy(buffer,num_actions,size_step),
                                                                           matrizQ(2, std::vector<double>(num_actions, 0.0)),
                                                                           matrizAlpha(2, std::vector<double>(num_actions, size_step)),
                                                                           matrizCounter(2, std::vector<int>(num_actions, 1))
{}

Bandit::~Bandit(){}

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

// Change the cost function

int Bandit::selectAction(int state){

    int action = -1;

    double randomNum = 0.0;

    if(training){
        randomNum = generateRandomDouble();
    }else{
        randomNum = 1.0;
    }

    logStream << "Ingreso a Seleccion de accion:" << std::endl;
    logStream << "EPSILON:" << epsilon << std::endl;

    if (randomNum > epsilon) {
        
        logStream << "Seleccion Greedy:" << std::endl;

        if(ruleta){
            logStream << "Usando ruleta:" << std::endl;

            // Selección basada en ruleta
            std::vector<double> valores = matrizQ[state];

/*
            std::cout << "Valores:"<< std::endl;

            //imprimir valores
            for (size_t i = 0; i < valores.size(); ++i) {
                std::cout << valores[i] << " ";
            }

            std::cout << std::endl;
*/
            double min_val = *std::min_element(valores.begin(), valores.end());

            // Desplazar todos los valores si hay negativos
            if (min_val < 0) {
                double ajuste = 1 - min_val;
                for (double& valor : valores) {
                    valor += ajuste;
                }
            }

/*
            std::cout << "Valores:"<< std::endl;

            //imprimir valores
            for (size_t i = 0; i < valores.size(); ++i) {
                std::cout << valores[i] << " ";
            }

            std::cout << std::endl;
*/
            // Sumar los valores para calcular la probabilidad acumulada
            double suma_total = std::accumulate(valores.begin(), valores.end(), 0.0);

            //std::cout << "Suma total:"<< suma_total << std::endl;

            double probabilidad_acumulada = 0.0;
            double ruleta_giro = generateRandomDouble() * suma_total;
            
            //std::cout << "Ruleta giro:"<< ruleta_giro << std::endl;

            for (size_t i = 0; i < valores.size(); ++i) {
                probabilidad_acumulada += valores[i];
                //std::cout << "Probabilidad acumulada:"<< probabilidad_acumulada << std::endl;
                if (ruleta_giro <= probabilidad_acumulada) {
                    action = i;
                    //std::cout << "Accion:"<< action << std::endl;
                    break;
                }
            }

        }else{

            logStream << "Seleccion Greedy determinista:" << std::endl;
            // Greedy selection
            std::vector<int> max_indices;
            double max_value = *std::max_element(matrizQ[state].begin(), matrizQ[state].end());

            // Travel matrizQ[state]
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

        }

        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    } else {
        // Random selection
        logStream << "Seleccion Aleatoria:" << std::endl;
        int randomAction = generateRandomInt(matrizQ[state].size());

        action = randomAction;
        logStream << "Accion escogida: "<<action<<std::endl<<std::endl;
    }

    //std::cout << "Accion escogida: "<<action<<std::endl;

    return action;
}

void Bandit::updateQ(int actual_state, int actual_action, double reward){

    // Travel matrizQ[state]
    logStream << std::endl << "Q Antes:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    if(loup_found > 1){
        //Significa que se encontro un loup, por lo tanto se usara un valor de alpha superior al original y luego se actualizara el valor de alpha, ademas de actualizar el contador.
        double alphaTemporal = 0.5;

        //Sera mas agresivo en acercarse a la recompensa obtenido (recompensa positiva), debido a que el alpha Temporal es mayor al alpha original.
        matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + alphaTemporal*(reward - matrizQ[actual_state][actual_action]);

        logStream<< std::endl << "Counter Antes:" << std::endl;
        for (size_t i = 0; i < matrizCounter[actual_state].size(); ++i) {
            logStream << matrizCounter[actual_state][i] << " ";
        }

        //El contador aumenta en 1 para esa posicion en la matriz. Esto hace que mientras mas veces encuentre buenas recompensas, mas lento cambiara el valor alto de Q, ya que el alpha sera menor.
        matrizCounter[actual_state][actual_action] = matrizCounter[actual_state][actual_action] + 1;

        logStream<< std::endl << "Counter Despues:" << std::endl;
        for (size_t i = 0; i < matrizCounter[actual_state].size(); ++i) {
            logStream << matrizCounter[actual_state][i] << " ";
        }

        logStream<< std::endl << "Alpha Antes:" << std::endl;
        for (size_t i = 0; i < matrizAlpha[actual_state].size(); ++i) {
            logStream << matrizAlpha[actual_state][i] << " ";
        }
        
        //Actualiza el valor de alpha para esa posicion en la matriz. Va disminuyendo el valor de alpha en el tiempo. Este valor de alpha solo se usa cuando no encuentra soluciones, en caso contrario utiliza el valor de alpha temporal.
        matrizAlpha[actual_state][actual_action] = alpha / matrizCounter[actual_state][actual_action];

        logStream<< std::endl << "Alpha Despues:" << std::endl;
        for (size_t i = 0; i < matrizAlpha[actual_state].size(); ++i) {
            logStream << matrizAlpha[actual_state][i] << " ";
        }

    }else{

        logStream<< std::endl << "Counter:" << std::endl;
        for (size_t i = 0; i < matrizCounter[actual_state].size(); ++i) {
            logStream << matrizCounter[actual_state][i] << " ";
        }

        logStream<< std::endl << "Alpha:" << std::endl;
        for (size_t i = 0; i < matrizAlpha[actual_state].size(); ++i) {
            logStream << matrizAlpha[actual_state][i] << " ";
        }

        matrizQ[actual_state][actual_action] = matrizQ[actual_state][actual_action] + matrizAlpha[actual_state][actual_action]*(reward - matrizQ[actual_state][actual_action]);  
    
    }  

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    logStream << std::endl<<"Q Despues:" << std::endl;
    for (size_t i = 0; i < matrizQ[actual_state].size(); ++i) {
        logStream << matrizQ[actual_state][i] << " ";
    }
}

void Bandit::updateQ(int actual_state, int future_state, int actual_action, int future_action, double reward){}


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
            //Epsilon decrese su valor en el tiempo, de manera que al principio explore mucho y luego explote mucho.
            decEpsilon();
        }
        
        change = true;
    }
}

void Bandit::decEpsilon(){
    //Reduciendo el epsilon en el tiempo con un exponencial decreciente.
    epsilon = epsilon * std::exp(-0.005 * actual_iteration);
    actual_iteration++;
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
            std::cout << elemento << "\t";
        }
        std::cout << std::endl;
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

        std::ofstream file("MatrixQBandit.txt");
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


void Bandit::loadVectorsFromFile() {
        std::ifstream file("MatrixQBandit.txt");
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

int Bandit::ActualState(bool searchType){
    
    int state = -1;

    if (!searchType) {
        // When searchType it's False (Explotación)
        state = 0;
    } else {
        // When searchType it's True (Exploración)
        state = 1;
    }

    return state;

}

int Bandit::ActualState(bool searchType, double width, int activeNodes, bool loupChange){return 0;}


void Bandit::StartExploration() {

    loadVectorsFromFile();

    logStream << "Start Exploration:" << std::endl;
    logStream << "Estado anterior: " <<estado_anterior<<std::endl;
    logStream << "Accion anterior: " <<accion_anterior<<std::endl;
    logStream << "Estado actual: " <<estado_actual<<std::endl;
    logStream << "Accion actual: " <<accion_actual<<std::endl<<std::endl;

    estado_actual = ActualState(true); // Exploration(1)
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
    std::ofstream file("Logs.txt", std::ios::app);
    if (file.is_open()) {
        file << logStream.str();
        logStream.str("");
        logStream.clear();
    } else {
        //std::cerr << "Failed to open the log file." << std::endl;
    }
}

int Bandit::getWidthCategory(double width) {
    return 0;
}

int Bandit::getActiveNodesCategory(int activeNodes) {
    return 0;
}

void Bandit::updateWidth(double loup, double uplo){}

void Bandit::setLoupChanged(bool loupChange){}

} // end namespace ibex
