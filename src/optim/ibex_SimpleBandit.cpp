#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdlib>  // Para rand() y srand()
#include <ctime>    // Para time()
#include <fstream>
#include "ibex_SimpleBandit.h"

namespace ibex {

//constructor
SimpleBandit::SimpleBandit(CellBeamSearch * buffer, int num_actions, double size_step) : buffer(buffer), k(num_actions), alpha(size_step){
    epsilon = 0.05;
    Q_exploration.resize(k, 0.0);
    Q_explotation.resize(k, 0.0);
    //N.resize(k, 0);
    change = false;
    nb_cells = 0;
    loup_variation = 0.0;
    action_exploration = 0;
    action_explotation = 0;
    update = false;
    srand(static_cast<unsigned>(time(0)));
}

SimpleBandit::~SimpleBandit(){
    if(buffer != nullptr){
        delete buffer;
    }
}

/*
void SimpleBandit::setLoupVariation(double variation){
    loup_variation = variation;
}

void SimpleBandit::setNbCells(int nb){
    nb_cells = nb;
}
*/

void SimpleBandit::resetVars(){
    nb_cells = 0;
    loup_variation = 0.0;
}

double SimpleBandit::calculateRewardExploration(){
    //return loup_variation;
    return -static_cast<double>(nb_cells);
}

double SimpleBandit::calculateRewardExplotation(){
    /*
    if(loup_variation == 0){
        return -static_cast<double>(nb_cells);
    }else{
        return -static_cast<double>(nb_cells)/loup_variation;
    }
    */
    return -static_cast<double>(nb_cells);
}

//Cambiar funcion de costo
void SimpleBandit::selectActionExploration(){
    //std::uniform_real_distribution<> prob(0.0, 1.0);
    //std::uniform_int_distribution<> action_selector(0, k-1);
    if (generateRandomDouble() > epsilon) {
        // Selección greedy
        action_exploration = std::distance(Q_exploration.begin(), std::max_element(Q_exploration.begin(), Q_exploration.end()));
    } else {
        // Selección aleatoria
        action_exploration = generateRandomInt();
    }
    if(buffer){
        buffer->setCost2Function(action_exploration);
        //std::cout << "Cambio exitoso Exploration" << std::endl;
    }
    
}

void SimpleBandit::selectActionExplotation(){
    //std::uniform_real_distribution<> prob(0.0, 1.0);
    //std::uniform_int_distribution<> action_selector(0, k-1);
    if (generateRandomDouble() > epsilon) {
        // Selección greedy
        action_explotation = std::distance(Q_explotation.begin(), std::max_element(Q_explotation.begin(), Q_explotation.end()));
    } else {
        // Selección aleatoria
        action_explotation = generateRandomInt();
    }
    if(buffer){
        buffer->setCost2Function(action_explotation);
        //std::cout << "Cambio exitoso Explotation" << std::endl;
    }
}

void SimpleBandit::updateQExploration(){
    Q_exploration[action_exploration] += (calculateRewardExploration() - Q_exploration[action_exploration]) * alpha;
}

void SimpleBandit::updateQExplotation(){
    Q_explotation[action_explotation] += (calculateRewardExplotation() - Q_explotation[action_explotation]) * alpha;
}

void SimpleBandit::MonitoringSize(){
    if(buffer->futurebuffer.size() == 0){
        if(update){
            //setLoupVariation(variationLoup);
            //setNbCells(num_cells);
            updateQExploration();
            updateQExplotation();
            resetVars();
            update=false;
            alpha = alpha/1.0001;
            //std::cout << "Alpha: " << alpha << std::endl;
        }
        selectActionExploration();
        change = true;
    }
}

void SimpleBandit::MonitoringChange(){
    if(change){
        selectActionExplotation();
        change = false;
        update = true;
    }
}

void SimpleBandit::printQ(){
    std::cout << "Q_exploration: ";
    for(int i = 0; i < Q_exploration.size(); i++){
        std::cout << Q_exploration[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Q_explotation: ";
    for(int i = 0; i < Q_explotation.size(); i++){
        std::cout << Q_explotation[i] << " ";
    }
    std::cout << std::endl;
}

void SimpleBandit::adder(bool loupChanged){
    nb_cells += 2;
    if(loupChanged){loup_variation += 1.0;}
}

double SimpleBandit::generateRandomDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int SimpleBandit::generateRandomInt() {
    return rand() % 8; // El rango es [0, 7], así que % 8
}

void SimpleBandit::saveVectorsToFile(){
    std::ofstream outFile("Q_Vectors.txt");
    if (outFile.is_open()) {
        // Guardar el tamaño de los vectores
        outFile << alpha << "\n";
        outFile << Q_exploration.size() << "\n";
        outFile << Q_explotation.size() << "\n";
        
        // Guardar los elementos del primer vector
        for (double element : Q_exploration) {
            outFile << element << " ";
        }
        outFile << "\n";
        
        // Guardar los elementos del segundo vector
        for (double element : Q_explotation) {
            outFile << element << " ";
        }
        outFile << "\n";
        
        outFile.close();
    } else {
        std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
    }
}

void SimpleBandit::loadVectorsFromFile() {
    std::ifstream inFile("Q_Vectors.txt");
    if (inFile.is_open()) {
        // Leer el tamaño de los vectores
        inFile >> alpha;
        size_t size1, size2;
        inFile >> size1 >> size2;
        
        Q_exploration.resize(size1);
        Q_explotation.resize(size2);
        
        // Leer los elementos del primer vector
        for (size_t i = 0; i < size1; ++i) {
            inFile >> Q_exploration[i];
        }
        
        // Leer los elementos del segundo vector
        for (size_t i = 0; i < size2; ++i) {
            inFile >> Q_explotation[i];
        }
        inFile.close();

        printQ();
    } else {
        std::cerr << "No se pudo abrir el archivo para lectura." << std::endl;
        printQ();
    }
}

} // end namespace ibex
