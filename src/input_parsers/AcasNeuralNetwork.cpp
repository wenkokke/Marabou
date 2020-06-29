/*********************                                                        */
/*! \file AcasNeuralNetwork.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "AcasNeuralNetwork.h"
#include "AcasNnet.h"
#include "DisjunctionConstraint.h"
#include "FloatUtils.h"
#include "Vector.h"

#include <iostream>

AcasNeuralNetwork::AcasNeuralNetwork( const String &path )
    : _network( NULL )
{
    _network = load_network( path.ascii() );
}

AcasNeuralNetwork::~AcasNeuralNetwork()
{
    if ( _network )
    {
        destroy_network( _network );
        _network = NULL;
    }
}

double AcasNeuralNetwork::getWeight( int sourceLayer, int sourceNeuron, int targetNeuron )
{
    return _network->matrix[sourceLayer][0][targetNeuron][sourceNeuron];
}

String AcasNeuralNetwork::getWeightAsString( int sourceLayer, int sourceNeuron, int targetNeuron )
{
    double weight = getWeight( sourceLayer, sourceNeuron, targetNeuron );
    return FloatUtils::doubleToString( weight );
}

String AcasNeuralNetwork::getBiasAsString( int layer, int neuron )
{
    double bias = getBias( layer, neuron );
    return FloatUtils::doubleToString( bias );
}

double AcasNeuralNetwork::getBias( int layer, int neuron )
{
    // The bias for layer i is in index i-1 in the array.
    assert( layer > 0 );
    return _network->matrix[layer - 1][1][neuron][0];
}

int AcasNeuralNetwork::getNumLayers() const
{
    return _network->numLayers;
}

unsigned AcasNeuralNetwork::getLayerSize( unsigned layer ) const
{
    return (unsigned)_network->layerSizes[layer];
}

void AcasNeuralNetwork::evaluate( const Vector<double> &inputs, Vector<double> &outputs, unsigned outputSize ) const
{
    Vector<double> inputVector( inputs.size(), 0.0 );
    Vector<double> outputVector( outputSize, 0.0 );
    double *input( inputVector.data() );
    double *output( outputVector.data() );

    for ( unsigned i = 0; i < inputs.size();  ++i )
        input[i] = inputs.get( i );

    bool normalizeInput = false;
    bool normalizeOutput = false;

    if ( evaluate_network( _network, input, output, normalizeInput, normalizeOutput ) != 1 )
    {
        std::cout << "Error! Network evaluation failed" << std::endl;
        exit( 1 );
    }

    for ( unsigned i = 0; i < outputSize; ++i )
    {
        outputs.append( output[i] );
    }
}

void AcasNeuralNetwork::getInputRange( unsigned index, double &min, double &max )
{
    max =
        ( _network->maxes[index] - _network->means[index] )
        / ( _network->ranges[index] );

    min =
        ( _network->mins[index] - _network->means[index] )
        / ( _network->ranges[index] );
}

PiecewiseLinearConstraint *AcasNeuralNetwork::getActivationFunction( int layer,
                                                                     int neuron,
                                                                     unsigned b,
                                                                     unsigned f )
{
    NLR::NeuronIndex index( layer - 1, neuron );

    if ( !_network->_activationInformation.exists( index ) )
    {
        printf( "Error! Activation function doesn't exist for <%u,%u>\n", layer, neuron );
        exit( 1 );
    }

    List<String> tokens = _network->_activationInformation[index].tokenize( "," );

    if ( tokens.size() % 4 != 0 )
    {
        printf( "Error! Malformed activation function string: %s\n",
                _network->_activationInformation[index].ascii() );
        exit( 1 );
    }

    List<PiecewiseLinearCaseSplit> splits;
    unsigned count = 0;
    auto it = tokens.begin();
    while ( count < tokens.size() )
    {
        count += 4;

        String lbString = *it++;
        String ubString = *it++;
        String coefficient = *it++;
        String scalar = *it++;

        PiecewiseLinearCaseSplit split;

        if ( lbString != "-infty" )
        {
            Tightening lb( b, atof( lbString.ascii() ), Tightening::LB );
            split.storeBoundTightening( lb );
        }

        if ( ubString != "infty" )
        {
            Tightening ub( b, atof( ubString.ascii() ), Tightening::UB );
            split.storeBoundTightening( ub );
        }

        Equation eq;
        // Text format: f = coef * b + scalar
        // Eq format: f - coef * b = scalar
        eq.addAddend( 1, f );
        eq.addAddend( -atof( coefficient.ascii() ), b );
        eq.setScalar( atof( scalar.ascii() ) );
        split.addEquation( eq );

        splits.append( split );
    }

    return new DisjunctionConstraint( splits );
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
