module ComplexBC

import JuMP
import MathOptInterface
import BranchAndBound
import PowerModels
import Mosek, MosekTools
import Ipopt
import LinearAlgebra: norm, rank

const MOI = MathOptInterface
const BB = BranchAndBound
const PM = PowerModels

const MOSEK_OPTIMIZER = JuMP.optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
const MAX_NODE_ALLOWED = 10000

include("utils.jl")
include("data_structures.jl")
include("pm_exts.jl")
include("key_steps.jl")
include("bb_exts.jl")
include("initialize.jl")

function main(file::String)::Nothing
    data = PM.parse_file(file)
    pm = PM.instantiate_model(data, NodeWRMPowerModel, PM.build_opf)
    optimizer = JuMP.optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    # optimizer = optimizer_with_attributes(SCS.Optimizer, "max_iters" => 10000, "verbose" => 0)
    JuMP.set_optimizer(pm.model, optimizer)

    tree, node = initialize(pm, file)

    @time BB.run(tree)

    if isempty(node.auxiliary_data["rank1"])
        println("No rank-1 solution found")
    else
        (bound, id) = findmin(node.auxiliary_data["rank1"])
        println("Best bound obtained at $(id), bound value $(bound)")
    end
end

end # module