/*
 * Quadrilateral.hpp
 *
 *  Created on: Apr 29, 2020
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_FACTORYPLUGIN_QUADRILATERAL_HPP_
#define EXAMPLES_SRC_FACTORYPLUGIN_QUADRILATERAL_HPP_
#include "FactoryTraits.hpp"
namespace Geometry
{
  /*!
   * Derived class for quadrilaterals
   */
  class Quadrilateral: public Polygon
  {
  public:
    Quadrilateral();
    explicit Quadrilateral(Vertices const & v);
    explicit Quadrilateral(Vertices const & v,bool convex);
    virtual ~Quadrilateral();
    virtual void showMe(ostream & out=cout) const override;
    // other stuff in the future...
  };

  //! Not for genral use
  namespace internals
  {
    //! Loads the factory when lib is loaded
    static void __attribute__ ((constructor)) LoadF();
  }

}




#endif /* EXAMPLES_SRC_FACTORYPLUGIN_QUADRILATERAL_HPP_ */
