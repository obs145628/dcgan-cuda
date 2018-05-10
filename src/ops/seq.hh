#pragma once

#include "op.hh"

namespace ops
{

    class Seq : public Op
    {

    public:
        /**
         * Execute a list of operations
         * It must not be empty
         * Its value is the last operation
         */
        Seq(std::vector<Op*> ops);

        virtual void compile() override;
        
    };
    
}
