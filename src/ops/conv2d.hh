#pragma once

#include "op.hh"

namespace ops
{

  class Conv2D : public Op
  {
  public:
    Conv2D(Op* input, Op* kernel, const int* strides);

    virtual void compile() override;
  private:
    const int* m_strides;
  };
}
