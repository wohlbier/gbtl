/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#include <iostream>
#include <limits>

#include <algorithms/apsp.hpp>
#include <graphblas/graphblas.hpp>

using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apsp_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
template <typename T>
GraphBLAS::Matrix<T> get_tn_answer()
{
    std::vector<GraphBLAS::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<GraphBLAS::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6,
        8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5,
        6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 7, 0, 1, 2, 3, 4, 5, 6, 8
    };

    std::vector<T> vals = {
        0,   2,   3,   1,   2,   4,   2,   3,   2, 0,   2,   1,   2,
        3,   1,   3,   3,   2, 0,   2,   1,   1,   1,   1,   1,   1,
        2, 0,   1,   3,   1,   2,   2,   2,   1,   1, 0,   2,   2,
        1,   4,   3,   1,   3,   2, 0,   2,   2,   2,   1,   1,   1,
        2,   2, 0,   2, 0,   3,   3,   1,   2,   1,   2,   2, 0};

    //std::vector<std::vector<T> > G_tn_answer_dense =
    //    {{0, 2, 3, 1, 2, 4, 2, , 3},
    //     {2, 0, 2, 1, 2, 3, 1, , 3},
    //     {3, 2, 0, 2, 1, 1, 1, , 1},
    //     {1, 1, 2, 0, 1, 3, 1, , 2},
    //     {2, 2, 1, 1, 0, 2, 2, , 1},
    //     {4, 3, 1, 3, 2, 0, 2, , 2},
    //     {2, 1, 1, 1, 2, 2, 0, , 2},
    //     { ,  ,  ,  ,  ,  ,  , 0, },
    //     {3, 3, 1, 2, 1, 2, 2, , 0}};
    GraphBLAS::Matrix<T> temp(9,9);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}

//****************************************************************************
template <typename T>
GraphBLAS::Matrix<T> get_gilbert_answer()
{
    //std::vector<std::vector<T> > G_gilbert_answer_dense =
    //    {{  0,   1,   2,   1,   2,   3,   2},
    //     {  3,   0,   2,   2,   1,   2,   1},
    //     {   ,    ,   0,    ,    ,   1,    },
    //     {  1,   2,   1,   0,   3,   2,   3},
    //     {   ,    ,   2,    ,   0,   1,    },
    //     {   ,    ,   1,    ,    ,   0,    },
    //     {  2,   3,   1,   1,   1,   2,   0}};
    //return Matrix<T>(G_gilbert_answer_dense, );
    std::vector<GraphBLAS::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    };
    std::vector<GraphBLAS::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    };
    std::vector<T> vals = {
        0,   1,   2,   1,   2,   3,   2,   3, 0,   2,   2,   1,   2,
        1, 0,   1,   1,   2,   1, 0,   3,   2,   3,   2, 0,   1,
        1, 0,   2,   3,   1,   1,   1,   2, 0
    };
    GraphBLAS::Matrix<T> temp(7,7);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(apsp_basic_double_batch)
{
    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    GraphBLAS::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto distances = apsp(G_tn);
    auto G_tn_answer(get_tn_answer<double>());

    BOOST_CHECK_EQUAL(distances, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apsp_basic_uint_batch)
{
    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto distances = apsp(G_tn);
    auto G_tn_answer(get_tn_answer<unsigned int>());

    BOOST_CHECK_EQUAL(distances, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apsp_gilbert_double_batch)
{
    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double>       v(i.size(), 1);
    GraphBLAS::Matrix<double> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);
    //print_matrix(std::cout, G_gilbert, "Graph");

    // Perform for all roots simultaneously
    auto distances = apsp(G_gilbert);
    auto G_gilbert_answer(get_gilbert_answer<double>());

    //print_matrix(std::cout, distances, "result");
    //print_matrix(std::cout, G_gilbert_answer, "correct answer");
    BOOST_CHECK_EQUAL(distances, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apsp_gilbert_uint_batch)
{
    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    auto distances = apsp(G_gilbert);
    auto G_gilbert_answer(get_gilbert_answer<unsigned int>());

    BOOST_CHECK_EQUAL(distances, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(new_apsp_gilbert_uint)
{
    unsigned int const INF = 666666;
    // The correct answer for all starting points (in order)
    std::vector<std::vector<unsigned int> > G_gilbert_answer_dense =
        {{  0,   1,   2,   1,   2,   3,   2},
         {  3,   0,   2,   2,   1,   2,   1},
         {INF, INF,   0, INF, INF,   1, INF},
         {  1,   2,   1,   0,   3,   2,   3},
         {INF, INF,   2, INF,   0,   1, INF},
         {INF, INF,   1, INF, INF,   0, INF},
         {  2,   3,   1,   1,   1,   2,   0}};
    GraphBLAS::Matrix<unsigned int>
        G_gilbert_answer(G_gilbert_answer_dense, INF);

    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i.begin(), j.begin(), v.begin(), i.size());

    auto G_gilbert_res = apsp(G_gilbert);

    BOOST_CHECK_EQUAL(G_gilbert_res, G_gilbert_answer);
}

BOOST_AUTO_TEST_SUITE_END()
