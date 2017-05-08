/*
 * Copyright (c) 2017 Carnegie Mellon University and The Trustees of
 * Indiana University.
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

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sparse_ewisemult_vector_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sparse_ewisemult_vector_suite)

//****************************************************************************

namespace
{
    std::vector<double> v3a_dense = {12, 0, 7};

    std::vector<double> v4a_dense = {0, 0, 12, 7};
    std::vector<double> v4b_dense = {0,  1, 0, 2};
    std::vector<double> v4c_dense = {3,  6, 9, 1};

    std::vector<double> zero3_dense = {0, 0, 0};
    std::vector<double> zero4_dense = {0, 0, 0, 0};

    std::vector<double> twos3_dense = {2, 2, 2};
    std::vector<double> twos4_dense = {2, 2, 2, 2};

    std::vector<double> ans_4atwos4_dense = {0, 0, 24, 14};
    std::vector<double> ans_4a4b_dense = {0, 0, 0, 14};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v3a_dense, 0.);
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    GraphBLAS::Vector<double> result(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException);

    // incompatible output vector dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), u, u)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_reg)
{
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_stored_zero_result)
{
    GraphBLAS::Vector<double> u(v4a_dense); //, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense); //, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    ans2.setElement(1, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}
#if 0
//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              Mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u,
                              true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_reg)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 5);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 6);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 5);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 6);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              Mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException)
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_reg)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 11);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 12);
        BOOST_CHECK_EQUAL(result, ans);
    }

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_reg_stored_zero)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 11);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 12);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::complement(Mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u,
                              true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_reg)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 5);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 6);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 5);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u,
                             true);

        BOOST_CHECK_EQUAL(result.nvals(), 6);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::complement(Mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException)
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_reg)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 11);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 12);
        BOOST_CHECK_EQUAL(result, ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Vector<double> v(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> u(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense);//, 0.);

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 11);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        GraphBLAS::Vector<double> result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), v, u);

        BOOST_CHECK_EQUAL(result.nvals(), 12);
        BOOST_CHECK_EQUAL(result, ans);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()