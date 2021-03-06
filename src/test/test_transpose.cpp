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

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <iostream>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE transpose_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_bad_dim)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(3, 4);

    BOOST_CHECK_THROW(
        (transpose(mC, NoMask(), NoAccumulate(), mA)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_square)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(j_mA, i_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(3, 3);

    transpose(mC, NoMask(), NoAccumulate(), mA);
    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_nonsquare)
{
    IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3,-2, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(4, 3);
    answer.build(j_mA, i_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(4, 3);

    transpose(mC, NoMask(), NoAccumulate(), mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose_square)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(j_mA, i_mA, v_mA);

    auto result = transpose(mA);

    IndexType r(result.nrows());
    IndexType c(result.ncols());

    BOOST_CHECK_EQUAL(r, 3);
    BOOST_CHECK_EQUAL(c, 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose_nonsquare)
{
    IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3,-2, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(4, 3);
    answer.build(j_mA, i_mA, v_mA);

    auto result = transpose(mA);

    IndexType r(result.nrows());
    IndexType c(result.ncols());

    BOOST_CHECK_EQUAL(r, 4);
    BOOST_CHECK_EQUAL(c, 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_transpose_noaccum)
{
    // Build input matrix
    // | 1 1 - - |T
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0},
                                             {1, 2, 2},
                                             {0, 2, 3},
                                             {0, 0, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    std::vector<std::vector<double>> Ctmp = {{9, 9, 9, 9},
                                             {9, 9, 9, 9},
                                             {9, 9, 9, 9}};

    std::vector<std::vector<uint8_t>> mask = {{1, 1, 1, 1},
                                              {0, 1, 1, 1},
                                              {0, 0, 0, 1}};
    Matrix<uint8_t> M(mask, 0);
    {
        std::vector<std::vector<double>> ans = {{1, 1, 0, 0},
                                                {1, 2, 2, 0},
                                                {0, 2, 3, 3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  NoMask(),
                  NoAccumulate(),
                  A);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{1, 1, 0, 0},
                                                {0, 2, 2, 0},
                                                {0, 0, 0, 3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  M,
                  NoAccumulate(),
                  A,
                  true);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{1, 1, 0, 0},
                                                {9, 2, 2, 0},
                                                {9, 9, 9, 3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  M,
                  NoAccumulate(),
                  A,
                  false);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // scmp(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{0, 0, 0, 0},
                                                {1, 0, 0, 0},
                                                {0, 2, 3, 0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  complement(M),
                  NoAccumulate(),
                  A,
                  true);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // scmp(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{9, 9, 9, 9},
                                                {1, 9, 9, 9},
                                                {0, 2, 3, 9}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  complement(M),
                  NoAccumulate(),
                  A,
                  false);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_transpose_plus_accum)
{
    // Build input matrix
    // | 1 1 - - |T
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0},
                                             {1, 2, 2},
                                             {0, 2, 3},
                                             {0, 0, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    std::vector<std::vector<double>> Ctmp = {{9, 9, 9, 9},
                                             {9, 9, 9, 9},
                                             {9, 9, 9, 9}};

    std::vector<std::vector<uint8_t>> mask = {{1, 1, 1, 1},
                                              {0, 1, 1, 1},
                                              {0, 0, 0, 1}};
    Matrix<uint8_t> M(mask, 0);
    {
        std::vector<std::vector<double>> ans = {{10, 10,  9,  9},
                                                {10, 11, 11,  9},
                                                { 9, 11, 12, 12}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  NoMask(),
                  Plus<double>(),
                  A);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{10, 10,  9,  9},
                                                {0,  11, 11,  9},
                                                {0,   0,  0, 12}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  M,
                  Plus<double>(),
                  A,
                  true);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{10, 10, 9,  9},
                                                { 9, 11, 11, 9},
                                                { 9, 9,  9, 12}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  M,
                  Plus<double>(),
                  A,
                  false);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // scmp(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{ 0,  0,  0, 0},
                                                {10,  0,  0, 0},
                                                { 9, 11, 12, 0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  complement(M),
                  Plus<double>(),
                  A,
                  true);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // scmp(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{ 9,  9,  9, 9},
                                                {10,  9,  9, 9},
                                                { 9, 11, 12, 9}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        transpose(C,
                  complement(M),
                  Plus<double>(),
                  A,
                  false);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_transpose_nonsquare_trans)
{
    // Build input matrix
    // | 1 1 - - |T
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0, 0},
                                             {1, 2, 2, 0},
                                             {0, 2, 3, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    {
        std::vector<std::vector<double>> ans = {{1, 1, 0, 0},
                                                {1, 2, 2, 0},
                                                {0, 2, 3, 3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(3, 4);
        transpose(C,
                  NoMask(),
                  NoAccumulate(),
                  transpose(A));
        BOOST_CHECK_EQUAL(C, answer);
    }
}

BOOST_AUTO_TEST_SUITE_END()
