/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#include <iostream>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sparse_extract_suite

#include <boost/test/included/unit_test.hpp>

// @todo:  Why do I have to do this??
#include <graphblas/system/sequential/sparse_extract.hpp>

BOOST_AUTO_TEST_SUITE(sparse_extract_suite)

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_base)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> matAnswer = {{1, 6},
                                                 {9, 2}};

    GraphBLAS::LilSparseMatrix<double> mA(matA, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 2;
    GraphBLAS::IndexType N = 2;

    GraphBLAS::LilSparseMatrix<bool> mask(M, N);

    GraphBLAS::IndexArrayType row_indicies = {0, 2};
    GraphBLAS::IndexArrayType col_indicies = {1, 2};

    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::extract(result,
                                mask,
                                GraphBLAS::Second<double>(),
                                mA,
                                row_indicies,
                                col_indicies);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_duplicate)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> matAnswer = {{1, 1, 6},
                                                  {9, 9, 2},
                                                  {9, 9, 2}};


    GraphBLAS::LilSparseMatrix<double> mA(matA, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;

    GraphBLAS::LilSparseMatrix<bool> mask(M, N);

    GraphBLAS::IndexArrayType row_indicies = {0, 2, 2};
    GraphBLAS::IndexArrayType col_indicies = {1, 1, 2};

    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::extract(result,
                                mask,
                                GraphBLAS::Second<double>(),
                                mA,
                                row_indicies,
                                col_indicies);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_permute)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> matAnswer = {{9, 2, 4},
                                                  {1, 6, 8},
                                                  {5, 7, 3}};


    GraphBLAS::LilSparseMatrix<double> mA(matA, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;

    GraphBLAS::LilSparseMatrix<bool> mask(M, N);

    GraphBLAS::IndexArrayType row_indicies = {2, 0, 1};
    GraphBLAS::IndexArrayType col_indicies = {1, 2, 0};

    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::extract(result,
                                mask,
                                GraphBLAS::Second<double>(),
                                mA,
                                row_indicies,
                                col_indicies);

    BOOST_CHECK_EQUAL(result, answer);
}


BOOST_AUTO_TEST_SUITE_END()