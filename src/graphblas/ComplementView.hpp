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

#pragma once

#include <cstddef>
#include <graphblas/Matrix.hpp>

#define GB_INCLUDE_BACKEND_COMPLEMENT_VIEW 1
#include <backend_include.hpp>

//****************************************************************************
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template<typename MatrixT>
    class MatrixComplementView
    {
    public:
        typedef typename backend::MatrixComplementView<
            typename MatrixT::BackendType> BackendType;
        typedef typename MatrixT::ScalarType ScalarType;

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        MatrixComplementView(BackendType backend_view)
            : m_mat(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        MatrixComplementView(MatrixComplementView<MatrixT> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }

        ~MatrixComplementView() { }

        /// @todo need to change to mix and match internal types
        template <typename OtherMatrixT>
        bool operator==(OtherMatrixT const &rhs) const
        {
            return (m_mat == rhs);
        }

        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
        }

        IndexType nrows() const { return m_mat.nrows(); }
        IndexType ncols() const { return m_mat.ncols(); }
        IndexType nvals() const { return m_mat.nvals(); }

        bool hasElement(IndexType row, IndexType col) const
        {
            return (m_mat.hasElement(row,col));
        }

        ScalarType extractElement(IndexType row, IndexType col) const
        {
            return m_mat.extractElement(row, col);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorJT,
                 typename RAIteratorVT,
                 typename AMatrixT>
        inline void extractTuples(RAIteratorIT        row_it,
                                  RAIteratorJT        col_it,
                                  RAIteratorVT        values)
        {
            m_mat.extractTuples(row_it, col_it, values);
        }

        template<typename ValueT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        inline void extractTuples(RowSequenceT              &row_indices,
                                  ColSequenceT              &col_indices,
                                  std::vector<ValueT>       &values)
        {
            m_mat.extractTuples(row_indices, col_indices, values);
        }

        //other methods that may or may not belong here:
        //
        void printInfo(std::ostream &os) const
        {
            os << "Frontend MatrixComplementView of:";
            m_mat.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixComplementView const &mat)
        {
            mat.printInfo(os);
            return os;
        }

    private:
        BackendType m_mat;

        // PUT ALL FRIEND DECLARATIONS (that use matrix masks) HERE

        // 4.3.1:
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        friend inline Info mxm(CMatrixT         &C,
                               MaskT      const &Mask,
                               AccumT            accum,
                               SemiringT         op,
                               AMatrixT   const &A,
                               BMatrixT   const &B,
                               bool              replace_flag);

        //--------------------------------------------------------------------

        // 4.3.4.2:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline Info eWiseMult(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.5.2
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline Info eWiseAdd(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.6.2
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename ...CTags>
        friend inline Info extract(
                GraphBLAS::Matrix<CScalarT, CTags...>   &C,
                MaskT               const   &Mask,
                AccumT                       accum,
                AMatrixT            const   &A,
                RowSequenceT        const   &row_indices,
                ColSequenceT        const   &col_indices,
                bool                         replace_flag);

        //--------------------------------------------------------------------
        // 4.3.7.2
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename std::enable_if<
                     std::is_same<matrix_tag,
                                  typename AMatrixT::tag_type>::value,
                     int>::type>
        friend inline Info assign(CMatrixT              &C,
                                  MaskT           const &Mask,
                                  AccumT                 accum,
                                  AMatrixT        const &A,
                                  RowSequenceT    const &row_indices,
                                  ColSequenceT    const &col_indices,
                                  bool                   replace_flag);

        // 4.3.7.6
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename std::enable_if<
                     std::is_convertible<ValueT,
                                         typename CMatrixT::ScalarType>::value,
                     int>::type>
        friend inline Info assign(CMatrixT             &C,
                                  MaskT          const &Mask,
                                  AccumT                accum,
                                  ValueT                val,
                                  RowSequenceT   const &row_indices,
                                  ColSequenceT   const &col_indices,
                                  bool                  replace_flag);

        //--------------------------------------------------------------------

        // 4.3.8.2
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename AMatrixT,
                 typename ...ATagsT>
        friend inline Info apply(
            GraphBLAS::Matrix<CScalarT, ATagsT...>           &C,
            MaskT                                     const  &Mask,
            AccumT                                            accum,
            UnaryFunctionT                                    op,
            AMatrixT                                  const  &A,
            bool                                              replace_flag);

        //--------------------------------------------------------------------

        // 4.3.10
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        friend inline Info transpose(CMatrixT       &C,
                                     MaskT    const &Mask,
                                     AccumT          accum,
                                     AMatrixT const &A,
                                     bool            replace_flag);

        //--------------------------------------------------------------------

    };

    //************************************************************************
    //************************************************************************
    template<typename VectorT>
    class VectorComplementView
    {
    public:
        typedef typename backend::VectorComplementView<
            typename VectorT::BackendType> BackendType;
        typedef typename VectorT::ScalarType ScalarType;

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        VectorComplementView(BackendType backend_view)
            : m_vec(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The vector to copy.
         */
        VectorComplementView(VectorComplementView<VectorT> const &rhs)
            : m_vec(rhs.m_vec)
        {
        }

        ~VectorComplementView() { }

        /// @todo need to change to mix and match internal types
        template <typename OtherVectorT>
        bool operator==(OtherVectorT const &rhs) const
        {
            return (m_vec == rhs);
        }

        template <typename OtherVectorT>
        bool operator!=(OtherVectorT const &rhs) const
        {
            return !(*this == rhs);
        }

        IndexType size() const  { return m_vec.size(); }
        IndexType nvals() const { return m_vec.nvals(); }

        bool hasElement(IndexType index) const
        {
            return m_vec.hasElement(index);
        }

        ScalarType extractElement(IndexType index) const
        {
            return m_vec.extractElement(index);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorVT>
        inline void extractTuples(RAIteratorIT        i_it,
                                  RAIteratorVT        v_it)
        {
            m_vec.extractTuples(i_it, v_it);
        }

        // @TODO: Should this be const Indices
        template<typename ValueT,
                 typename SequenceT>
        inline void extractTuples(SequenceT                 &indices,
                                  std::vector<ValueT>       &values)
        {
            m_vec.extractTuples(indices, values);
        }

        //other methods that may or may not belong here:
        //
        void printInfo(std::ostream &os) const
        {
            os << "Frontend VectorComplementView of:";
            m_vec.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream               &os,
                                        VectorComplementView const &vec)
        {
            vec.printInfo(os);
            return os;
        }

    private:
        BackendType m_vec;

        // PUT ALL FRIEND DECLARATIONS (that use vector masks) HERE

        //--------------------------------------------------------------------
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        friend inline Info vxm(WVectorT         &w,
                               MaskT      const &mask,
                               AccumT            accum,
                               SemiringT         op,
                               UVectorT   const &u,
                               AMatrixT   const &A,
                               bool              replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        friend inline Info mxv(WVectorT        &w,
                               MaskT     const &mask,
                               AccumT           accum,
                               SemiringT        op,
                               AMatrixT  const &A,
                               UVectorT  const &u,
                               bool             replace_flag);

        //--------------------------------------------------------------------

        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline Info eWiseMult(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            bool                                    replace_flag);

        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline Info eWiseAdd(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.6.1
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        friend inline Info extract(WVectorT             &w,
                                   MaskT          const &mask,
                                   AccumT                accum,
                                   UVectorT       const &u,
                                   SequenceT      const &indices,
                                   bool                  replace_flag);

        // 4.3.6.3
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename SequenceT,
                 typename ...WTags>
        friend inline Info extract(
                GraphBLAS::Vector<WScalarT, WTags...> &w,
                MaskT          const &mask,
                AccumT                accum,
                AMatrixT       const &A,
                SequenceT      const &row_indices,
                IndexType             col_index,
                bool                  replace_flag);

        //--------------------------------------------------------------------

        // 4.3.7.1: assign - standard vector variant
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename std::enable_if<
                     std::is_same<vector_tag,
                                  typename UVectorT::tag_type>::value,
                     int>::type,
                 typename ...WTags>
        friend inline void assign(Vector<WScalarT, WTags...>      &w,
                                  MaskT                     const &mask,
                                  AccumT                           accum,
                                  UVectorT                  const &u,
                                  SequenceT                 const &indices,
                                  bool                             replace_flag);

        // 4.3.7.3: assign - column variant
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline void assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                       accum,
                                  UVectorT              const &u,
                                  SequenceT             const &row_indices,
                                  IndexType                    col_index,
                                  bool                         replace_flag);

        // 4.3.7.4: assign - row variant
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline void assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                       accum,
                                  UVectorT              const &u,
                                  IndexType                    row_index,
                                  SequenceT             const &col_indices,
                                  bool                         replace_flag);

        // 4.3.7.5:
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename SequenceT,
                 typename std::enable_if<
                     std::is_convertible<ValueT,
                                         typename WVectorT::ScalarType>::value,
                     int>::type>
        friend inline Info assign(WVectorT &w,
                                  MaskT const &mask,
                                  AccumT accum,
                                  ValueT val,
                                  SequenceT const &indices,
                                  bool replace_flag);

        //--------------------------------------------------------------------
        // 4.3.8.1:
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename UVectorT,
                 typename ...WTagsT>
        friend inline Info apply(GraphBLAS::Vector<WScalarT, WTagsT...> &w,
                                 MaskT                            const &mask,
                                 AccumT                                  accum,
                                 UnaryFunctionT                          op,
                                 UVectorT                         const &u,
                                 bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.9.1
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline Info reduce(WVectorT        &w,
                                  MaskT     const &mask,
                                  AccumT           accum,
                                  BinaryOpT        op,
                                  AMatrixT  const &A,
                                  bool             replace_flag);
    };

} // end namespace GraphBLAS
