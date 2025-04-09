//
//  main.cpp
//  NeuralNetwork
//
//  Created by Anton Haas on 2020. 07. 04..
//  Copyright © 2020. Anton Haas. All rights reserved.
//


 //2 hidden layers
//#include <iostream>
//#include <math.h>

#define layers 4
#define inputnodes 3
#define hiddennodes 5
#define hiddennodes2 5
#define outputnodes 2

double stepsize = 0.1;

struct node{
    double input;
    double output; //input after sigmoid
};

double input [inputnodes];
node hidden [hiddennodes];
node hidden2 [hiddennodes2];
node output [outputnodes];

double weights2 [inputnodes][hiddennodes];
double weights3 [hiddennodes][hiddennodes2];
double weights4 [hiddennodes2][outputnodes];


double weightgradients2 [inputnodes][hiddennodes];
double weightgradients3 [hiddennodes][hiddennodes2];
double weightgradients4 [hiddennodes2][outputnodes];

double outputgradients2[hiddennodes];
double outputgradients3[hiddennodes2];
double outputgradients4[outputnodes];

double target[outputnodes];

double b2 [hiddennodes];
double b3 [hiddennodes2];
double b4 [outputnodes];

double error;

double sigmoid(double x) {
    return (1.0f / (1.0f + exp(-x)));
}

double sigmoidderivative(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

void initializeweightsandbiases(){
    int x;
    for (int i = 0; i < hiddennodes2; i++)
    {
        for (int j = 0; j < outputnodes; j++)
        {
            x = rand() % 100 + 1;
            weights4[i][j] = (double)1/(double)x;
            
            x = rand() % 100 + 1;
            b3[i] = (double)1/(double)x;
        }
    }
    for (int i = 0; i < hiddennodes; i++)
    {
        for (int j = 0; j < hiddennodes2; j++)
        {
            x = rand() % 100 + 1;
            weights3[i][j] = (double)1/(double)x;
            
            x = rand() % 100 + 1;
            b3[i] = (double)1/(double)x;
        }
    }
    for (int i = 0; i < inputnodes; i++)
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            x = rand() % 100 + 1;
            weights2[i][j] = (double)1/(double)x;
            
            x = rand() % 100 + 1;
            b2[i] = (double)1/(double)x;
        }
    }
}

void reset(){
    for (int i = 0; i < inputnodes; i++)
    {
        input[i] = 0;
    }
    for (int i = 0; i < hiddennodes; i++)
    {
        hidden[i].input = 0;
        hidden[i].output = 0;
    }
    for (int i = 0; i < hiddennodes2; i++)
    {
        hidden2[i].input = 0;
        hidden2[i].output = 0;
    }
    for (int i = 0; i < outputnodes; i++)
    {
        output[i].input = 0;
        output[i].output = 0;
    }
}

void feedforward(double input1, double input2, double input3) {

    reset();
    
    input[0] = input1;
    input[1] = input2;
    input[2] = input3;
    
    for (int i = 0; i < hiddennodes; i++)
    {
        for (int j = 0; j < inputnodes; j++)
        {
            hidden[i].input += input[j] * weights2[j][i];
        }
        hidden[i].input += b2[i];
        hidden[i].output = sigmoid(hidden[i].input);
    }
    
    for (int i = 0; i < hiddennodes2; i++)
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            hidden2[i].input += hidden[j].output * weights3[j][i];
        }
        hidden2[i].input += b3[i];
        hidden2[i].output = sigmoid(hidden2[i].input);
    }
    
    for (int i = 0; i < outputnodes; i++)
    {
        for (int j = 0; j < hiddennodes2; j++)
        {
            output[i].input += hidden2[j].output * weights4[j][i];
        }
        output[i].input += b4[i];
        output[i].output = sigmoid(output[i].input);
    }
}

void backpropagate()
{
    for (int i = 0; i < outputnodes; i++) // last layer weights and outputgradients
    {
        outputgradients4[i] = (2 * (output[i].output - target[i])); // sigmoid'(input) * activationgradient
        for (int j = 0; j < hiddennodes2; j++)
        {
            weightgradients4[j][i] = hidden2[j].output * sigmoidderivative(output[i].input) * outputgradients4[i];
        }
    }
    
    for (int i = 0; i < hiddennodes2; i++) // hidden2 outputgradients
    {
        outputgradients3[i] = 0;
        for (int j = 0; j < outputnodes; j++)
        {
            outputgradients3[i] += weights4[i][j] * sigmoidderivative(output[j].input) * outputgradients4[j];
        }
    }
    
    for (int i = 0; i < hiddennodes2; i++) // hidden2 weightsgradients
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            weightgradients3[j][i] = hidden[j].output * sigmoidderivative(hidden2[i].input) * outputgradients3[i];
        }
    }
    
    for (int i = 0; i < hiddennodes; i++) // hidden layer outputgradients
    {
        outputgradients2[i] = 0;
        for (int j = 0; j < hiddennodes2; j++)
        {
            outputgradients2[i] += weights3[i][j] * sigmoidderivative(hidden2[j].input) * outputgradients3[j];
        }
    }
    
    for (int i = 0; i < hiddennodes; i++) // hidden layer weightgradients
    {
        for (int j = 0; j < inputnodes; j++)
        {
            weightgradients2[j][i] = input[j] * sigmoidderivative(hidden[i].input) * outputgradients2[i];
        }
    }
    
    
    for (int i = 0; i < outputnodes; i++) // updating
    {
        for (int j = 0; j < hiddennodes2; j++)
        {
            weights4[j][i] -= stepsize * weightgradients4[j][i]; // updating weights
        }
        b4[i] -= stepsize * sigmoidderivative(output[i].input) * outputgradients3[i];
    }
    for (int i = 0; i < hiddennodes2; i++) // updating
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            weights3[j][i] -= stepsize * weightgradients3[j][i]; // updating weights
        }
        b3[i] -= stepsize * sigmoidderivative(hidden2[i].input) * outputgradients3[i];
    }
    for (int i = 0; i < hiddennodes; i++) // hidden layer weights
    {
        for (int j = 0; j < inputnodes; j++)
        {
            weights2[j][i] -= stepsize * weightgradients2[j][i];
        }
        b2[i] -= stepsize * sigmoidderivative(hidden[i].input) * outputgradients2[i];
    }
}

double testinput[3] = {24,6,48};

int main(int argc, const char * argv[])
{
    initializeweightsandbiases ();
    target[0] = (double)0.5;
    target[1] = (double)0.6;
    for(int i = 0; i < 1000; i++)
    {
        feedforward(testinput[0], testinput[1], testinput[2]);
        
        error = (target[0]-output[0].output) * (target[0]-output[0].output) + (target[1]-output[1].output) * (target[1]-output[1].output);
        
        std::cout<< output[0].output << " " << output[1].output << " " << error <<std::endl;
        
        backpropagate();
    }
        
    return 0;
}

/*
 //one hidden layer
 
#include <iostream>
#include <math.h>

#define layers 3
#define inputnodes 3
#define hiddennodes 30
#define outputnodes 2

double stepsize = 0.05;

struct node{
    double input;
    double output; //input after sigmoid
};

double input [inputnodes];
node hidden [hiddennodes];
node output [outputnodes];

double weights2 [inputnodes][hiddennodes];
double weights3 [hiddennodes][outputnodes];


double weightgradients2 [inputnodes][hiddennodes];
double weightgradients3 [hiddennodes][outputnodes];

double outputgradients2[hiddennodes];
double outputgradients3[outputnodes];

double target[2];

double b2 [hiddennodes];
double b3 [outputnodes];

double error;

double sigmoid(double x) {
    return (1.0f / (1.0f + exp(-x)));
}

double sigmoidderivative(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

void initializeweightsandbiases(){
    int x;
    for (int i = 0; i < hiddennodes; i++)
    {
        for (int j = 0; j < outputnodes; j++)
        {
            x = rand() % 100 + 1;
            weights3[i][j] = (double)1/(double)x;
            
            x = rand() % 100 + 1;
            b3[i] = (double)1/(double)x;
        }
    }
    for (int i = 0; i < inputnodes; i++)
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            x = rand() % 100 + 1;
            weights2[i][j] = (double)1/(double)x;
            
            x = rand() % 100 + 1;
            b2[i] = (double)1/(double)x;
        }
    }
}

void reset(){
    for (int i = 0; i < inputnodes; i++)
    {
        input[i] = 0;
    }
    for (int i = 0; i < hiddennodes; i++)
    {
        hidden[i].input = 0;
        hidden[i].output = 0;
    }
    for (int i = 0; i < outputnodes; i++)
    {
        output[i].input = 0;
        output[i].output = 0;
    }
}

void feedforward(double input1, double input2, double input3) {

    reset();
    
    input[0] = input1;
    input[1] = input2;
    input[2] = input3;
    
    for (int i = 0; i < hiddennodes; i++)
    {
        for (int j = 0; j < inputnodes; j++)
        {
            hidden[i].input += input[j] * weights2[j][i];
        }
        hidden[i].input += b2[i];
        hidden[i].output = sigmoid(hidden[i].input);
    }
    
    for (int i = 0; i < outputnodes; i++)
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            output[i].input += hidden[j].output * weights3[j][i];
        }
        output[i].input += b3[i];
        output[i].output = sigmoid(output[i].input);
    }
}

void backpropagate()
{
    for (int i = 0; i < outputnodes; i++) // last layer weights and outputgradients
    {
        outputgradients3[i] = (2 * (output[i].output - target[i])); // sigmoid'(input) * activationgradient
        for (int j = 0; j < hiddennodes; j++)
        {
            weightgradients3[j][i] = hidden[j].output * sigmoidderivative(output[i].input) * outputgradients3[i];
        }
    }
    
    for (int i = 0; i < hiddennodes; i++) // hidden layer outputgradients
    {
        outputgradients2[i] = 0;
        for (int j = 0; j < outputnodes; j++)
        {
            outputgradients2[i] += weights3[i][j] * sigmoidderivative(output[j].input) * outputgradients3[j];
        }
    }
    
    for (int i = 0; i < hiddennodes; i++) // hidden layer weights
    {
        for (int j = 0; j < inputnodes; j++)
        {
            weightgradients2[j][i] = input[j] * sigmoidderivative(hidden[i].input) * outputgradients2[i];
        }
    }
    
    
    for (int i = 0; i < outputnodes; i++) // updating
    {
        for (int j = 0; j < hiddennodes; j++)
        {
            weights3[j][i] -= stepsize * weightgradients3[j][i]; // updating weights
        }
        b3[i] -= stepsize * sigmoidderivative(output[i].input) * outputgradients3[i];
    }
    for (int i = 0; i < hiddennodes; i++) // hidden layer weights
    {
        for (int j = 0; j < inputnodes; j++)
        {
            weights2[j][i] -= stepsize * weightgradients2[j][i];
        }
        b2[i] -= stepsize * sigmoidderivative(hidden[i].input) * outputgradients2[i];
    }
}

double testinput[3] = {24,6,4};

int main(int argc, const char * argv[])
{
    initializeweightsandbiases ();
    target[0] = (double)0.8;
    target[1] = (double)0.9;
    for(int i = 0; i < 100; i++)
    {
        feedforward(testinput[0], testinput[1], testinput[2]);
        
        error = (target[0]-output[0].output) * (target[0]-output[0].output) + (target[1]-output[1].output) * (target[1]-output[1].output);
        
        std::cout<< output[0].output << " " << output[1].output << " " << error <<std::endl;
        
        backpropagate();
    }
        
    return 0;
}
*/
